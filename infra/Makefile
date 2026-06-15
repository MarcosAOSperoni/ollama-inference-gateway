include .env

start:
	envsubst < prometheus.yml.template > prometheus.yml
	docker compose up -d

stop:
	docker compose down

restart: stop start
