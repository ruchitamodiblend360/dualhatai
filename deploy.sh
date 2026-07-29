#!/bin/bash
# Push local changes to the live EC2 instance and restart the service.
#
# Usage:
#   ./deploy.sh              # sync code + static assets, restart service
#   ./deploy.sh server.py    # sync only the given file(s), restart service
#
# Requires: dualhatai-key.pem in this directory (chmod 400 on it).
set -euo pipefail

# git-bash/MSYS mangles "host:/path" colons into path lists — disable that.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

HOST="ec2-user@50.16.167.48"
KEY="dualhatai-key.pem"
APPDIR="/opt/dualhatai"

if [ ! -f "$KEY" ]; then echo "Missing $KEY in $(pwd)"; exit 1; fi
chmod 400 "$KEY" 2>/dev/null || true

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(server.py index.html landing.html status.html board.html vendor \
         blend-logo-white.png Blend360-Logo.png cover-bg.jpeg \
         team-francisco.jpeg team-naida.png team-ruchita.png team-ruchita.jpg \
         team-ruchita.jfif team-ruchita-updated.jfif user_story_readiness_checker_2.jsx)
fi

echo ">> copying to instance /tmp ..."
ssh -i "$KEY" "$HOST" "rm -rf /tmp/deploy && mkdir -p /tmp/deploy"
scp -i "$KEY" -r "${FILES[@]}" "$HOST:/tmp/deploy/"

echo ">> installing + restarting ..."
ssh -i "$KEY" "$HOST" \
  "sudo cp -r /tmp/deploy/* $APPDIR/ && sudo systemctl restart dualhatai && sleep 1 && systemctl is-active dualhatai"

echo ">> done. http://50.16.167.48/"
