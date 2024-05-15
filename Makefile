run:
	@xhost +
	@docker compose run --rm sitl /usr/bin/bash

build:
	@docker compose build

