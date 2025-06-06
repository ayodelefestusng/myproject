#!/bin/sh
set -e

# Wait for PostgreSQL
echo "Waiting for PostgreSQL at $DATABASE_HOST:$DATABASE_PORT..."
while ! nc -z $DATABASE_HOST $DATABASE_PORT; do
  sleep 2
done
echo "PostgreSQL is ready!"

# Django setup
python manage.py migrate --noinput
python manage.py collectstatic --noinput --clear

exec "$@"