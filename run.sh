python3 aulm.py --model_name lenet_aulm_1 \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--learning_rate_decay_type fixed\
				--max_number_of_steps 8000 \
				--update_fy_step 200 \
				--update_layer_name conv1 \
				--checkpoint_path model/lenet.ckpt \
				--train_dir tmp/lenet_aulm_conv1 

python3 train.py --model_name lenet_aulm_1_ft \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--num_epochs_per_decay 20\
				--max_number_of_steps 12000 \
				--checkpoint_path tmp/lenet_aulm_conv1/model.ckpt-8000 \
				--train_dir tmp/lenet_aulm_conv1_ft

python3 aulm.py --model_name lenet_aulm_2 \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--learning_rate_decay_type fixed\
				--max_number_of_steps 8000 \
				--update_fy_step 200 \
				--update_layer_name conv2 \
				--checkpoint_path tmp/lenet_aulm_conv1_ft/model.ckpt-12000 \
				--train_dir tmp/lenet_aulm_conv2 
				
python3 train.py --model_name lenet_aulm_2_ft \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--num_epochs_per_decay 20\
				--max_number_of_steps 12000 \
				--checkpoint_path tmp/lenet_aulm_conv2/model.ckpt-8000 \
				--train_dir tmp/lenet_aulm_conv2_ft
				
python3 aulm.py --model_name lenet_aulm_3 \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--learning_rate_decay_type fixed\
				--max_number_of_steps 8000 \
				--update_fy_step 200 \
				--update_layer_name fc1 \
				--checkpoint_path tmp/lenet_aulm_conv2_ft/model.ckpt-12000 \
				--train_dir tmp/lenet_aulm_fc1 

python3 train.py --model_name lenet_aulm_3_ft \
				--preprocessing_name lenet \
				--train_image_size 28 \
				--batch_size 256 \
				--learning_rate 0.01 \
				--num_epochs_per_decay 20\
				--max_number_of_steps 12000 \
				--checkpoint_path tmp/lenet_aulm_fc1/model.ckpt-8000 \
				--train_dir tmp/lenet_aulm_fc1_ft

python3 eval.py --model_name lenet_aulm_3_ft \
				--preprocessing_name lenet \
				--eval_image_size 28 \
				--batch_size 100 \
				--checkpoint_path tmp/lenet_aulm_fc1_ft/model.ckpt-12000 \
