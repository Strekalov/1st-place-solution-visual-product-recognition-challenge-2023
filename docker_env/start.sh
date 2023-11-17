
docker run --gpus all -it --ipc=host \
    -v /mnt/:/mnt/ \
    -v /home/cv_user/visual-product-recognition-2023-giga-flex:/home/cv_user/visual-product-recognition-2023-giga-flex \
    -v /home/cv_user/clean_thresh_dataset:/home/cv_user/clean_thresh_dataset \
    gigafastenv:latest /bin/bash
