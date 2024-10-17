export HF_ENDPOINT="https://hf-mirror.com"

INPUT_STY_DIR='./data/input/style'
INPUT_CNT_DIR='./data/input/content'
OUTPUT_IMG_DIR='./data/output'
GPU_CNT=8
MAX_NUM=8

for ((folder_number = 8; folder_number <= $MAX_NUM; folder_number += $GPU_CNT));do
    for((gpu_id = 0; gpu_id < GPU_CNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        CUDA_VISIBLE_DEVICES=$gpu_id

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_si.py \
            --sty_idx=$(printf "%02d" $current_folder_number)"

        eval $COMMAND &
        sleep 3
    done
    wait
done