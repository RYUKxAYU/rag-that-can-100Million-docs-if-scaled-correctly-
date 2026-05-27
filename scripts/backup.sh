#!/usr/bin/env sh
set -eu

BACKUP_SOURCE="${BACKUP_SOURCE:-/app/data}"
BACKUP_DEST="${BACKUP_DEST:-/app/backups}"
RETENTION_COUNT="${RETENTION_COUNT:-7}"

mkdir -p "$BACKUP_SOURCE" "$BACKUP_DEST"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_DEST/backup-$timestamp.tar.gz"

tar -czf "$archive" -C "$BACKUP_SOURCE" .

echo "Created backup archive: $archive"

count=0
for file in $(find "$BACKUP_DEST" -maxdepth 1 -type f -name 'backup-*.tar.gz' | sort -r); do
  count=$((count + 1))
  if [ "$count" -gt "$RETENTION_COUNT" ]; then
    rm -f "$file"
  fi
done
