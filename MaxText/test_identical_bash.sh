compile_dump_dir=$1
train_dump_dir=$2

export LIBTPU_INIT_ARGS=--xla_tpu_spmd_rng_bit_generator_unsafe=true
export XLA_FLAGS=--xla_dump_to=${compile_dump_dir}
python3 MaxText/train_compile.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs run_name=compile_equivalent_test dataset_path=gs://maxtext-dataset dataset_type=synthetic steps=5 enable_checkpointing=False compile_topology=v4-8 compile_topology_num_slices=1 quantization=int8    

export XLA_FLAGS=--xla_dump_to=${train_dump_dir}
python MaxText/train.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs run_name=compile_equivalent_test_real dataset_path=gs://maxtext-dataset dataset_type=synthetic steps=5 enable_checkpointing=False quantization=int8

