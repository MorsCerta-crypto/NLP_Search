
FROM mongo

WORKDIR /home

# Copy seed data
COPY . .

# Restore data
CMD sleep 5; mongorestore --host mongodb --drop --gzip --dir ./seeds;
