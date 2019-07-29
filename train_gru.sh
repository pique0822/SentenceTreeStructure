for hidden in 100 90 80 70 60 50 40 30 20 10
do
	python3 gru_arithmetic.py --hidden_size $hidden --num_epochs 100
done

# sbatch --time 100:0:0 --gres=gpu bash_train.sh
#14236473
