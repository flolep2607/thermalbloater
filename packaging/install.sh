#!/bin/sh
# Install thermalbloater so it heats the room inside a daily window, and keeps itself
# up to date from GitHub Releases.
#
#   curl -fsSL https://raw.githubusercontent.com/flolep2607/thermalbloater/main/packaging/install.sh | sudo sh
#   sudo ./packaging/install.sh                          # 17:00 -> 22:00, default args
#   sudo START=20 END=23 ./packaging/install.sh           # 20:00 -> 23:00
#   sudo ARGS="--gpu-max 70 --math f16" ./packaging/install.sh
#   sudo BIN=./target/release/thermalbloater ./install.sh # use a local build, skip the download
#   sudo NO_UPDATE=1 ./packaging/install.sh               # skip the daily self-update job
#   sudo UNINSTALL=1 ./packaging/install.sh               # remove everything
#
# Uses systemd when it is running (start timer + hard stop via RuntimeMaxSec), and falls
# back to crontab otherwise (start line + pkill line), so WSL and containers work too.
set -eu

REPO=flolep2607/thermalbloater
URL=https://github.com/$REPO/releases/latest/download/thermalbloater-linux-x86_64
BIN_PATH=/usr/local/bin/thermalbloater
UPDATE_PATH=/usr/local/bin/thermalbloater-update
UNIT_DIR=/etc/systemd/system

START=${START:-17}
END=${END:-22}
ARGS=${ARGS:-}
BIN=${BIN:-}

[ "$(id -u)" = 0 ] || { echo "run me as root (sudo)" >&2; exit 1; }
have_systemd() { [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; }

# Cron lines carry this marker so uninstall can strip exactly what we added.
crontab_without_ours() { crontab -l 2>/dev/null | grep -v '# thermalbloater$' || true; }

if [ -n "${UNINSTALL:-}" ]; then
    if have_systemd; then
        systemctl disable --now thermalbloater.timer thermalbloater-update.timer 2>/dev/null || true
        rm -f "$UNIT_DIR"/thermalbloater.service "$UNIT_DIR"/thermalbloater.timer \
              "$UNIT_DIR"/thermalbloater-update.service "$UNIT_DIR"/thermalbloater-update.timer
        systemctl daemon-reload
    fi
    command -v crontab >/dev/null 2>&1 && crontab_without_ours | crontab -
    pkill -x thermalbloater 2>/dev/null || true
    rm -f "$BIN_PATH" "$UPDATE_PATH"
    echo "thermalbloater removed."
    exit 0
fi

[ "$END" -gt "$START" ] || { echo "END ($END) must be later than START ($START)" >&2; exit 1; }

# Never write over the running binary in place (ETXTBSY): stage next to it and rename,
# which swaps the inode atomically and leaves any running instance on the old one.
install_bin() {
    install -m 755 "$1" "$2.new"
    mv "$2.new" "$2"
}

if [ -z "$BIN" ] && [ -x ./target/release/thermalbloater ]; then
    BIN=./target/release/thermalbloater
fi
if [ -n "$BIN" ]; then
    [ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }
    install_bin "$BIN" "$BIN_PATH"
else
    echo "Downloading the latest release..."
    tmp=$(mktemp)
    trap 'rm -f "$tmp"' EXIT
    curl -fsSL "$URL" -o "$tmp"
    install_bin "$tmp" "$BIN_PATH"
fi

# Self-update: fetch the latest asset and swap it in only if the bytes differ. No version
# compare, no API call, no jq — a few MiB a day is cheaper than parsing JSON in sh.
cat > "$UPDATE_PATH" <<EOF
#!/bin/sh
set -eu
tmp=\$(mktemp)
trap 'rm -f "\$tmp"' EXIT
curl -fsSL "$URL" -o "\$tmp"
cmp -s "\$tmp" "$BIN_PATH" && exit 0
install -m 755 "\$tmp" "$BIN_PATH.new"
mv "$BIN_PATH.new" "$BIN_PATH"
echo "thermalbloater updated; the new binary is used from the next run."
EOF
chmod 755 "$UPDATE_PATH"
[ -n "${NO_UPDATE:-}" ] && rm -f "$UPDATE_PATH"

if have_systemd; then
    cat > "$UNIT_DIR/thermalbloater.service" <<EOF
[Unit]
Description=thermalbloater (GPU space heater)
After=network.target

[Service]
ExecStart=$BIN_PATH $ARGS
# Hard stop at the end of the window. The timer starts us at $START:00.
RuntimeMaxSec=$(( (END - START) * 3600 ))
Nice=19
Restart=no
EOF

    cat > "$UNIT_DIR/thermalbloater.timer" <<EOF
[Unit]
Description=Run thermalbloater from ${START}:00 to ${END}:00 daily

[Timer]
OnCalendar=*-*-* ${START}:00:00
# ponytail: no Persistent= — booting mid-window skips that evening rather than
# firing a late run that would then overrun END. Add a stop-timer if that bites.

[Install]
WantedBy=timers.target
EOF

    if [ -x "$UPDATE_PATH" ]; then
        cat > "$UNIT_DIR/thermalbloater-update.service" <<EOF
[Unit]
Description=Update thermalbloater from GitHub Releases

[Service]
Type=oneshot
ExecStart=$UPDATE_PATH
EOF
        cat > "$UNIT_DIR/thermalbloater-update.timer" <<EOF
[Unit]
Description=Daily thermalbloater update check

[Timer]
OnCalendar=daily
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF
    fi

    systemctl daemon-reload
    systemctl enable --now thermalbloater.timer
    [ -x "$UPDATE_PATH" ] && systemctl enable --now thermalbloater-update.timer
    systemctl list-timers 'thermalbloater*' --no-pager
else
    command -v crontab >/dev/null 2>&1 || {
        echo "no systemd and no crontab — install one, or run $BIN_PATH yourself" >&2; exit 1; }
    {
        crontab_without_ours
        echo "0 $START * * * $BIN_PATH $ARGS >>/var/log/thermalbloater.log 2>&1 # thermalbloater"
        echo "0 $END * * * pkill -x thermalbloater # thermalbloater"
        [ -x "$UPDATE_PATH" ] && echo "30 4 * * * $UPDATE_PATH >>/var/log/thermalbloater.log 2>&1 # thermalbloater"
    } | crontab -
    echo "No systemd here — installed as cron jobs instead:"
    crontab -l | grep '# thermalbloater$'
fi
