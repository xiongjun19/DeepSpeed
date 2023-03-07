
cnt_name=$1
ssh_port=2222
docker run --name ${cnt_name} -it   --gpus all --ipc=host -p ${ssh_port}:${ssh_port}  -v /local/workspace/jxiong/workspace/data/:/workspace/data -v /nfs/homes/jxiong/workspace/DeepSpeed:/workspace/deepspeed  deepspeed /bin/bash
