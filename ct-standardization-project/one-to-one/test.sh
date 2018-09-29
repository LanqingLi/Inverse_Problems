python main.py --dataset_path '/home/boya/Data/CT/val' --phase train --gpu 0 --style_A 'guangzhou_yifuyi/dcm' --style_B 'STANDARD/dcm'

python main.py --dataset_path '/home/boya/Data/CT/val' --phase test --gpu 0 --style_A 'guangzhou_yifuyi/dcm' --style_B 'STANDARD/dcm' --out_path 'guangzhou_yifuyi_to_STANDARD/dcm'

cp -a '/home/boya/Data/CT/val/guangzhou_yifuyi_to_STANDARD/anno' '/home/boya/Data/CT/val/guangzhou_yifuyi_to_STANDARD/anno'

cd ..
cd ct_frcnn_detection
python predict_demo.py --image_dir '/home/boya/Data/CT/val/guangzhou_yifuyi_to_STANDARD/dcm'  --prefix 'model/test_model' --epoch 12 --multi --dicom --gpu 0 --auto

cd ..
cd CT-style-tranfer

rm -r /home/boya/Data/CT/val/checkpoints



python main.py --dataset_path '/home/boya/Data/CT/val' --phase train --gpu 0 --style_A 'Liaocheng_75/dcm' --style_B 'STANDARD/dcm'

python main.py --dataset_path '/home/boya/Data/CT/val' --phase test --gpu 0 --style_A 'Liaocheng_75/dcm' --style_B 'STANDARD/dcm' --out_path 'Liaocheng_75_to_STANDARD/dcm'

cp -a '/home/boya/Data/CT/val/Liaocheng_75/anno' '/home/boya/Data/CT/val/Liaocheng_75_to_STANDARD/anno'

cd ..
cd ct_frcnn_detection
python predict_demo.py --image_dir '/home/boya/Data/CT/val/Liaocheng_75_to_STANDARD/dcm'  --prefix 'model/test_model' --epoch 12 --multi --dicom --gpu 0 --auto

cd ..
cd CT-style-tranfer

rm -r /home/boya/Data/CT/val/checkpoints
