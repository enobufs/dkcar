DKCAR_IP_ADDRESS := enobufs-donkey-car.local

.PHONY: pull-data push-model clean-tub drive

pull-data:
	rsync -r pi@$(DKCAR_IP_ADDRESS):~/dkcar/data/ ./data/

push-model:
	rsync -r ./models/ pi@$(DKCAR_IP_ADDRESS):~/dkcar/models/

clean-tub:
	rm -f tub/*.jpg tub/record_*.json

drive:
	python manage.py drive
