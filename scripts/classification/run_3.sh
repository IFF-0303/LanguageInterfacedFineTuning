#run_3.sh

#Generate PGD from MLP 

#python -m lift.experiments.classification.utils.prepare_mnist -a 1 -e 0.1 --source mlp --target lenet
#python -m lift.experiments.classification.utils.prepare_mnist -a 1 -e 0.01 --source mlp --target lenet


# python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -u 1 -e 0.01 > run_gptj_mnist_noise_0_01 2>&1
# python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -u 1 -e 0.1 > run_gptj_mnist_noise_0_1 2>&1
# python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -u 1 -e 0.3 > run_gptj_mnist_noise_0_3 2>&1


#python -m lift.experiments.classification.utils.prepare_mnist -n -t const -e 0.001
#python -m lift.experiments.classification.utils.prepare_mnist -n -t const -e 0.3
#python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -m 100 -d mnist -v 1 -n 1 -t const -e 0.001 #> run_gptj_mnist_constant_noise_0_01 2>&1

#python -m lift.experiments.classification.utils.prepare_mnist -n -t const -e 0.01
#python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -t const -e 0.01 > run_gptj_mnist_constant_noise_0_01_with_10000_samples 2>&1

#export CUDA_VISIBLE_DEVICES=1
#python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -t const -e 0.1 > run_gptj_mnist_constant_noise_0_1_with_10000_samples 2>&1
#python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -v 1 -n 1 -t const -e 0.3 > run_gptj_mnist_constant_noise_0_3_with_10000_samples 2>&1
#python -m lift.experiments.classification.utils.prepare_mnist -n -t const -e 0.1
#python -m lift.experiments.classification.utils.prepare_mnist -n -t const -e 0.3


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -m 100 -v 1 -n 1 -t normal -e 0.01

#export CUDA_VISIBLE_DEVICES=1
# python -m lift.experiments.classification.utils.prepare_mnist -n -t normal -e 0.02
# python -m lift.experiments.classification.run_exps.run_gptj_mnist_perturbed -d mnist -m 100 -v 1 -n 1 -t normal -e 0.02 #> run_gptj_mnist_normal_noise_0_01_with_10000_samples 2>&1

#python -m lift.experiments.classification.run_exps.run_gpt3_mnist -n -t normal -e 0.01 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noisy_normal_0_01 2>&1 &
#python -m lift.experiments.classification.run_exps.run_gpt3_mnist -n -t normal -e 0.02 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noisy_normal_0_02 2>&1 &
#python -m lift.experiments.classification.run_exps.run_gpt3_mnist -n -t normal -e 0.05 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noisy_normal_0_05 2>&1 &
# python -m lift.experiments.classification.run_exps.run_gpt3_mnist -n -t normal -e 0.1 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noisy_normal_0_1_rerun 2>&1 &
# python -m lift.experiments.classification.run_exps.run_gpt3_mnist -n -t normal -e 0.3 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noisy_normal_0_3 2>&1 &

