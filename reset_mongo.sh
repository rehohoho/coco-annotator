#remove all docker container
docker rm $(docker ps -a -q)

#remove mongo volume
docker volume rm coco-annotator_mongodb_data
