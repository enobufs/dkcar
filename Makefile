DKCAR_IP_ADDRESS := enobufs-donkey-car.local

.PHONY: pull-data push-model

pull-data:
	rsync -r pi@$(DKCAR_IP_ADDRESS):~/dkcar/data/ ./data/

push-model:
	rsync -r ./models/ pi@$(DKCAR_IP_ADDRESS):~/dkcar/models/
