for hidden in 100 90 80 70 60 50 40 30 20 10
do
	python3 gru_arithmetic.py --hidden_size $hidden --num_epochs 100
done

# sbatch --mem 32 --time 1000 run_training.sh
