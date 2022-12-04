
#!/bin/bash
# declare an array called array and define 3 vales

# Loop over all the possible scenarios here
dataset=( drone-polysemi drone )
char_embed=( lstm cnn adatrans )
word_embed=( elmo glove bert )
output_dir="output-2"
attention_type=( cosine adatrans transformer bilstm )
scaled_attention=( true false )
counter=( 1 2 3 )
for dataset in "${dataset[@]}"
do
    for char_embed in "${char_embed[@]}"
    do
        for word in "${word_embed[@]}"
        do
            for attention in "${attention_type[@]}"
            do
                if [ "$attention" = "bilstm" ];
                then
                    for count in "${counter[@]}"
                    do
                        python train_bilstm.py --dataset "$dataset" --char_embed "$char_embed" --word_embed "$word" --counter "$count" --output_dir output
                    done
                else
                    for scaled in "${scaled_attention[@]}"
                    do
                        for count in "${counter[@]}"
                        do
                            if [ "$scaled" = "true" ];
                            then
                                python train_tener_en.py --dataset "$dataset" --char_embed "$char_embed" --word_embed "$word" --attention_type "$attention" --scaled_attention --counter "$count" --output_dir output
                            else
                                python train_tener_en.py --dataset "$dataset" --char_embed "$char_embed" --word_embed "$word" --attention_type "$attention" --counter "$count" --output_dir output
                            fi
                            # case "$count" in
                            # true) scaled="--scaled_attention";;
                            # false) a="" ;;
                            # esac

                        done
                    done
                fi
            done
        done
    done
done