#!/bin/bash

# setup script for container monitor
# run this once to set everything up

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/container-monitor"
LOG_DIR="$HOME/container-monitor-logs"
SERVICE_DIR="$HOME/.config/systemd/user"

echo "╔════════════════════════════════════════════════╗"
echo "║  Container Monitor Setup                       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# check if monitor.sh exists in repo
if [[ ! -f "$REPO_DIR/monitor.sh" ]]; then
    echo "Error: monitor.sh not found in $REPO_DIR"
    echo "Make sure monitor.sh is in the same directory as this setup script."
    exit 1
fi

echo "Repository directory: $REPO_DIR"
echo "Install directory: $INSTALL_DIR"
echo "Log directory: $LOG_DIR"
echo ""

# create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$SERVICE_DIR"

# copy monitor.sh from repo to install directory
echo "Installing monitor.sh..."
cp "$REPO_DIR/monitor.sh" "$INSTALL_DIR/monitor.sh"
chmod +x "$INSTALL_DIR/monitor.sh"

# verify email is configured
if grep -q 'ALERT_EMAIL="your-team@example.com"' "$INSTALL_DIR/monitor.sh"; then
    echo ""
    echo "WARNING: Email address not configured!"
    echo "Please edit $INSTALL_DIR/monitor.sh and set ALERT_EMAIL to your email address."
    echo ""
fi

# create systemd user service
echo "Creating systemd service..."
cat > "$SERVICE_DIR/container-monitor.service" <<EOF
[Unit]
Description=Container Monitor for goopy-app
After=default.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/monitor.sh monitor
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/monitor.log
StandardError=append:$LOG_DIR/monitor-error.log

[Install]
WantedBy=default.target
EOF

# create helper scripts

echo "Creating helper scripts..."

# 1. check status script
cat > "$INSTALL_DIR/check-status.sh" <<EOF
#!/bin/bash
$INSTALL_DIR/monitor.sh status
EOF
chmod +x "$INSTALL_DIR/check-status.sh"

# 2. view reports script
cat > "$INSTALL_DIR/view-reports.sh" <<EOF
#!/bin/bash
$INSTALL_DIR/monitor.sh report
EOF
chmod +x "$INSTALL_DIR/view-reports.sh"

# 3. analyze crash script
cat > "$INSTALL_DIR/analyze-crash.sh" <<EOF
#!/bin/bash
$INSTALL_DIR/monitor.sh analyze
EOF
chmod +x "$INSTALL_DIR/analyze-crash.sh"

# 4. generate summary report
cat > "$INSTALL_DIR/summary-report.sh" <<'EOF'
#!/bin/bash

LOG_DIR="$HOME/container-monitor-logs"
DOWNTIME_LOG="$LOG_DIR/downtime.log"
CRASH_REPORT="$LOG_DIR/crash-reports.log"

echo "==================================="
echo "Container Monitoring Summary Report"
echo "Generated: $(date)"
echo "==================================="
echo ""

# count crashes
if [[ -f "$CRASH_REPORT" ]]; then
    crash_count=$(grep -c "^CRASH REPORT$" "$CRASH_REPORT" 2>/dev/null || echo 0)
    echo "Total Crashes: $crash_count"
else
    echo "Total Crashes: 0"
fi

# calculate total downtime
if [[ -f "$DOWNTIME_LOG" ]]; then
    total_seconds=0
    while IFS= read -r line; do
        if [[ $line =~ Duration:.*\(([0-9]+)\ seconds\) ]]; then
            total_seconds=$((total_seconds + ${BASH_REMATCH[1]}))
        fi
    done < "$DOWNTIME_LOG"

    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))

    printf "Total Downtime: %02d:%02d:%02d (%d seconds)\n" $hours $minutes $seconds $total_seconds
else
    echo "Total Downtime: 00:00:00 (0 seconds)"
fi

echo ""
echo "==================================="
echo "Recent Downtime Events (Last 10)"
echo "==================================="

if [[ -f "$DOWNTIME_LOG" ]]; then
    tail -n 50 "$DOWNTIME_LOG" | grep -A 2 "Down Since" | tail -n 30
else
    echo "No downtime events recorded"
fi

echo ""
echo "==================================="
echo "Log Locations"
echo "==================================="
echo "Alerts: $LOG_DIR/alerts.log"
echo "Downtime: $LOG_DIR/downtime.log"
echo "Crashes: $LOG_DIR/crash-reports.log"
echo "Monitor: $LOG_DIR/monitor.log"
EOF
chmod +x "$INSTALL_DIR/summary-report.sh"

# 5. email test script
cat > "$INSTALL_DIR/test-email.sh" <<'EOF'
#!/bin/bash

# extract email from monitor.sh
MONITOR_SCRIPT="$HOME/container-monitor/monitor.sh"
ALERT_EMAIL=$(grep '^ALERT_EMAIL=' "$MONITOR_SCRIPT" | cut -d'"' -f2)

if [[ -z "$ALERT_EMAIL" ]] || [[ "$ALERT_EMAIL" == "your-team@example.com" ]]; then
    echo "Error: Email not configured in monitor.sh"
    echo "Please edit $MONITOR_SCRIPT and set ALERT_EMAIL"
    exit 1
fi

echo "Testing email alert..."
echo "Sending to: $ALERT_EMAIL"
echo ""

echo "This is a test alert from container monitoring system.

If you receive this email, alerts are working correctly

Sent at: $(date)
From server: $(hostname)" | mail -s "Test Alert - Container Monitor" "$ALERT_EMAIL"

if [[ $? -eq 0 ]]; then
    echo "Email sent successfully to $ALERT_EMAIL"
    echo "Check your inbox (and spam folder)"
else
    echo "Failed to send email. Check if mail command is configured."
    echo ""
    echo "To configure mail on this system, you may need to:"
    echo "  1. Install mailx: sudo yum install mailx"
    echo "  2. Configure sendmail or postfix"
    echo "  3. Or use an external SMTP relay"
fi
EOF
chmod +x "$INSTALL_DIR/test-email.sh"

# 6. auto-restart script
cat > "$INSTALL_DIR/auto-restart.sh" <<'EOF'
#!/bin/bash

CONTAINER_NAME="goopy-app"
LOG_DIR="$HOME/container-monitor-logs"

if ! docker ps --filter "name=$CONTAINER_NAME" | grep -q "$CONTAINER_NAME"; then
    echo "[$(date)] Container is down, attempting restart..." >> "$LOG_DIR/auto-restart.log"

    docker start "$CONTAINER_NAME" >> "$LOG_DIR/auto-restart.log" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[$(date)] Container restarted successfully" >> "$LOG_DIR/auto-restart.log"
    else
        echo "[$(date)] Failed to restart container" >> "$LOG_DIR/auto-restart.log"
    fi
fi
EOF
chmod +x "$INSTALL_DIR/auto-restart.sh"

# 7. dashboard script
cat > "$INSTALL_DIR/dashboard.sh" <<'EOF'
#!/bin/bash

LOG_DIR="$HOME/container-monitor-logs"
STATUS_FILE="$LOG_DIR/status.json"

while true; do
    clear
    echo "╔════════════════════════════════════════════════╗"
    echo "║     Container Monitoring Dashboard             ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # current status
    if docker ps --filter "name=goopy-app" | grep -q "goopy-app"; then
        echo "Status: RUNNING"
        echo ""
        docker ps --filter "name=goopy-app" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo "Status: DOWN"
        echo ""

        if [[ -f "$STATUS_FILE" ]]; then
            down_since=$(grep -o '"down_since":"[^"]*"' "$STATUS_FILE" | cut -d'"' -f4)
            if [[ -n "$down_since" ]]; then
                echo "Down since: $down_since"

                # Calculate downtime
                down_timestamp=$(date -d "$down_since" +%s 2>/dev/null)
                if [[ -n "$down_timestamp" ]]; then
                    now=$(date +%s)
                    duration=$((now - down_timestamp))
                    hours=$((duration / 3600))
                    minutes=$(((duration % 3600) / 60))
                    seconds=$((duration % 60))
                    printf "Duration: %02d:%02d:%02d\n" $hours $minutes $seconds
                fi
            fi
        fi
    fi

    echo ""
    echo "Press Ctrl+C to exit"
    echo "Refreshing every 5 seconds..."

    sleep 5
done
EOF
chmod +x "$INSTALL_DIR/dashboard.sh"

# 8. update script (re-run to update from repo)
cat > "$INSTALL_DIR/update-from-repo.sh" <<EOF
#!/bin/bash

REPO_DIR="$REPO_DIR"
INSTALL_DIR="$INSTALL_DIR"

echo "Updating monitor from repository..."

if [[ ! -f "\$REPO_DIR/monitor.sh" ]]; then
    echo "Error: monitor.sh not found in \$REPO_DIR"
    exit 1
fi

echo "Stopping service..."
systemctl --user stop container-monitor

echo "Copying updated monitor.sh..."
cp "\$REPO_DIR/monitor.sh" "\$INSTALL_DIR/monitor.sh"
chmod +x "\$INSTALL_DIR/monitor.sh"

echo "Restarting service..."
systemctl --user start container-monitor

echo "Update complete!"
systemctl --user status container-monitor
EOF
chmod +x "$INSTALL_DIR/update-from-repo.sh"

# reload systemd and enable service
echo ""
echo "Enabling systemd service..."
systemctl --user daemon-reload
systemctl --user enable container-monitor.service
systemctl --user start container-monitor.service

# enable lingering
echo "Enabling user lingering (allows service to run when logged out)..."
loginctl enable-linger $USER

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Setup Complete                                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# check service status
if systemctl --user is-active --quiet container-monitor; then
    echo "Monitoring service is running"
else
    echo "Warning: Service may not have started correctly"
    echo "Run: systemctl --user status container-monitor"
fi

echo ""
echo "Available commands:"
echo "  $INSTALL_DIR/check-status.sh       - Check current container status"
echo "  $INSTALL_DIR/view-reports.sh       - View all crash reports"
echo "  $INSTALL_DIR/summary-report.sh     - Generate summary report"
echo "  $INSTALL_DIR/analyze-crash.sh      - Analyze most recent crash"
echo "  $INSTALL_DIR/test-email.sh         - Test email alerts"
echo "  $INSTALL_DIR/dashboard.sh          - Live monitoring dashboard"
echo "  $INSTALL_DIR/update-from-repo.sh   - Update from repository"
echo ""
echo "Service management:"
echo "  systemctl --user status container-monitor    - Check service status"
echo "  systemctl --user stop container-monitor      - Stop monitoring"
echo "  systemctl --user restart container-monitor   - Restart monitoring"
echo "  systemctl --user logs container-monitor      - View service logs"
echo ""
echo "Logs location: $LOG_DIR"
echo ""

# check email configuration
if grep -q 'ALERT_EMAIL="your-team@example.com"' "$INSTALL_DIR/monitor.sh"; then
    echo "   IMPORTANT: Configure your email address"
    echo "   Edit: $INSTALL_DIR/monitor.sh"
    echo "   Change ALERT_EMAIL to your actual email"
    echo "   Then run: systemctl --user restart container-monitor"
else
    CONFIGURED_EMAIL=$(grep '^ALERT_EMAIL=' "$INSTALL_DIR/monitor.sh" | cut -d'"' -f2)
    echo "Email alerts configured for: $CONFIGURED_EMAIL"
    echo ""
    echo "Test email with: $INSTALL_DIR/test-email.sh"
fi

echo ""
echo "Repository: $REPO_DIR"
echo "Installation: $INSTALL_DIR"
