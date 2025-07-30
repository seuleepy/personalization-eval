#!/bin/bash

export WORKING_DIR="/path/to/your/working/directory"

filename=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" "cat" \
"cat2" "clock" "colorful_sneaker" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie" \
"monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie")

gen_folders=("gen-img-v1" "gen-img-v2" "gen-img-v3" "gen-img-v4" "gen-img-v5" "gen-img-v6" "gen-img-v7" "gen-img-v8")


total=${#filename[@]}
for gen_folder in "${gen_folders[@]}"; do

  for ((i=0; i<total; i++)); do
    file_item=${filename[$i]}

    echo "Evaluate #$(($i+1)) in $gen_folder $file_item"

    accelerate launch evaluation/eval_file_name.py \
      --gen_img_path="${WORKING_DIR}/${gen_folder}/${file_item}" \
      --real_img_path="${WORKING_DIR}/real/img/path/${file_item}" \
      --batch=50 \
      --output_path="/output/path/${gen_folder}/eval.json" \
      --seed=42

    if [ $? -ne 0 ]; then
      echo "Error: An error occurred while evaluating $gen_folder $file_item."
      exit 1
    fi

  done
done