nvidia-docker run -ti \
-v $(pwd):/workspace/ \
--ipc=host \
--net=host \
--name=$1 \
stylegan2ada \
/bin/bash
