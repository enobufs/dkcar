DKCAR_IP_ADDRESS := enobufs-donkey-car.local

.PHONY: pull-data push-model clean-tub drive docker-build docker-shell

pull-data:
	rsync -r pi@$(DKCAR_IP_ADDRESS):~/dkcar/data/ ./data/

push-model:
	rsync -r ./models/ pi@$(DKCAR_IP_ADDRESS):~/dkcar/models/

clean-tub:
	rm -f tub/*.jpg tub/record_*.json

drive:
	python manage.py drive

docker-build:
	docker build -t donkeychainer -f docker/docker-v4.3.0-intel-python3 .

docker-shell:
	docker run -it --rm -v$$(pwd):/home/donkey/dkcar donkeychainer:latest /bin/bash
