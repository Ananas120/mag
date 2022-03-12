USER_ID		= $(shell id -u)
USER_NAME	= $(shell id -u --name)
GROUP_ID	= $(shell id -u --name)

run:
	docker-compose -f docker-compose-maggie.yml up

build:
	docker-compose -f docker-compose-maggie.yml build --build-arg USER_ID=$(USER_ID) --build-arg USER_NAME=$(USER_NAME) --build-arg GROUP_ID=$(GROUP_ID)

down:
	docker-compose -f docker-compose-maggie.yml down

clean:
	rm -rf pretrained_models/**/training-logs/checkpoints/
	rm -rf pretrained_models/m4_*/saving/ckpt-1.*
	rm -rf pretrained_models/m4_*/saving/ckpt-2.*
	rm -rf pretrained_models/m5_*/saving/ckpt-1.*
	rm -rf pretrained_models/m5_*/saving/ckpt-2.*

