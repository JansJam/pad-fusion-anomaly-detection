import csv,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from pad.model import Features,PADPredictor,grid_boxes,inputs,combine_scores,validate_settings,nearest

ROOT=Path(__file__).resolve().parents[1]

class ContractTests(unittest.TestCase):
    def test_geometry_and_pixels(self):
        a=np.random.default_rng(1).integers(0,256,(385,387,3),dtype=np.uint8)
        im=Image.fromarray(a)
        boxes=grid_boxes(*im.size)
        self.assertEqual(boxes[0],(1,0,129,128))
        self.assertEqual(boxes[-1],(257,256,385,384))
        self.assertEqual(tuple(inputs(im,'native').shape),(9,3,128,128))
        self.assertEqual(tuple(inputs(im,'whole').shape),(1,3,256,256))
        with self.assertRaises(ValueError):grid_boxes(383,1024)

    def test_fusion_and_boundary(self):
        s={'rule':'maximum','scalers':{'whole':{'median':.5,'iqr':.1},'native':{'median':.5,'iqr':.2}},
           'thresholds':{'fusion':2.}}
        r=combine_scores(.5,.9,s)
        self.assertAlmostEqual(r['attack_score'],2.)
        self.assertEqual(r['predicted_attack'],0)
        self.assertEqual(r['dominant_branch'],'native')
        s['scalers']['native']['iqr']=0
        with self.assertRaises(ValueError):validate_settings(s)

    def test_nearest_against_explicit_distances(self):
        q=torch.tensor([[0.,1.],[1.,2.],[2.,3.]])
        b=torch.tensor([[0.,0.],[1.,2.]])
        expected=((q[:,None]-b[None,:])**2).sum(-1).sqrt().min(1).values
        self.assertTrue(torch.allclose(nearest(q,b,query_chunk=1,bank_chunk=1),expected))

    def test_cli_success_and_invalid_image(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);data=d/'images';data.mkdir();(data/'nested').mkdir()
            Image.new('RGB',(384,384),(80,120,140)).save(data/'valid.png')
            Image.new('RGB',(128,128)).save(data/'small.png')
            (data/'nested'/'broken.png').write_bytes(b'not an image')
            settings={'rule':'maximum','scalers':{'whole':{'median':.5,'iqr':.1},'native':{'median':.5,'iqr':.2}},
                      'thresholds':{'fusion':2.}}
            bank=torch.nn.functional.normalize(torch.randn(8,384),dim=1)
            modelpath=d/'test_model.pt'
            torch.save({'format_version':1,'kind':'whole_native_max_fusion','settings':settings,
                        'backbone_state':Features().state_dict(),'banks':{'whole':bank,'native':bank}},modelpath)
            expected=PADPredictor(modelpath).predict(data/'valid.png')
            cmd=[sys.executable,str(ROOT/'infer.py'),'--input',str(data),'--output',str(d/'scores.csv'),
                 '--model',str(modelpath),'--recursive']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            with (d/'scores.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual([r['image_id'] for r in rows],['nested/broken.png','small.png','valid.png'])
            for row in rows[:2]:
                self.assertEqual(row['status'],'error');self.assertEqual(row['attack_score'],'')
            self.assertAlmostEqual(float(rows[-1]['attack_score']),expected['attack_score'],places=5)
            self.assertEqual(int(rows[-1]['predicted_attack']),expected['predicted_attack'])
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,1)

if __name__=='__main__':unittest.main()
