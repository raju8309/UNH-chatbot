#!/bin/bash

# configuration
CONTAINER_NAME="goopy-app"
CHECK_INTERVAL=60
LOG_DIR="$HOME/container-monitor-logs"
ALERT_LOG="$LOG_DIR/alerts.log"
DOWNTIME_LOG="$LOG_DIR/downtime.log"
CRASH_REPORT="$LOG_DIR/crash-reports.log"
STATUS_FILE="$LOG_DIR/status.json"

# alert configuration
ALERT_EMAIL="" # fill in after cloning
ALERT_WEBHOOK="" # fill in after cloning

# initialize
mkdir -p "$LOG_DIR"
touch "$ALERT_LOG" "$DOWNTIME_LOG" "$CRASH_REPORT" "$STATUS_FILE"

# initialize crash report file with proper structure if it doesn't exist or is empty
if [[ ! -s "$CRASH_REPORT" ]]; then
    echo "=== Container Crash Reports ===" > "$CRASH_REPORT"
    echo "Generated: $(date)" >> "$CRASH_REPORT"
    echo "" >> "$CRASH_REPORT"
fi

# helper Functions

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ALERT_LOG"
}

get_container_status() {
    docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" 2>/dev/null
}

is_container_running() {
    local status=$(get_container_status)
    [[ -n "$status" ]]
}

get_container_id() {
    docker ps -aq --filter "name=$CONTAINER_NAME" 2>/dev/null | head -n1
}

analyze_crash_cause() {
    local container_id=$1
    local crash_reason="Unknown"
    local error_details=""

    if [[ -z "$container_id" ]]; then
        crash_reason="Container ID not found - may have been removed"
        return
    fi

    # get container exit code
    local exit_code=$(docker inspect "$container_id" --format='{{.State.ExitCode}}' 2>/dev/null)

    # get container state information
    local state_info=$(docker inspect "$container_id" --format='{{.State.Status}} | {{.State.Error}}' 2>/dev/null)

    # get last 100 lines of logs to analyze
    local logs=$(docker logs --tail 100 "$container_id" 2>&1)

    # analyze exit code
    case "$exit_code" in
        0)
            crash_reason="Clean exit (exit code 0) - Container stopped normally"
            ;;
        1)
            crash_reason="Application error (exit code 1)"
            ;;
        125)
            crash_reason="Docker daemon error (exit code 125)"
            ;;
        126)
            crash_reason="Container command cannot be invoked (exit code 126)"
            ;;
        127)
            crash_reason="Container command not found (exit code 127)"
            ;;
        137)
            crash_reason="Container killed by SIGKILL (exit code 137) - Out of memory or manual kill"
            ;;
        139)
            crash_reason="Segmentation fault (exit code 139)"
            ;;
        143)
            crash_reason="Container stopped by SIGTERM (exit code 143) - Graceful shutdown requested"
            ;;
        *)
            if [[ -n "$exit_code" ]]; then
                crash_reason="Unexpected exit (exit code $exit_code)"
            else
                crash_reason="Exit code unavailable"
            fi
            ;;
    esac

    # analyze logs for common error patterns
    if echo "$logs" | grep -qi "out of memory\|OOM\|oom"; then
        error_details="${error_details}• OUT OF MEMORY detected in logs\n"
    fi

    if echo "$logs" | grep -qi "panic\|fatal"; then
        error_details="${error_details}• PANIC/FATAL error detected in logs\n"
    fi

    if echo "$logs" | grep -qi "connection refused\|cannot connect"; then
        error_details="${error_details}• CONNECTION issues detected in logs\n"
    fi

    if echo "$logs" | grep -qi "permission denied"; then
        error_details="${error_details}• PERMISSION issues detected in logs\n"
    fi

    if echo "$logs" | grep -qi "no such file\|file not found"; then
        error_details="${error_details}• FILE NOT FOUND errors detected in logs\n"
    fi

    if echo "$logs" | grep -qi "port.*already in use\|address already in use"; then
        error_details="${error_details}• PORT CONFLICT detected in logs\n"
    fi

    if echo "$logs" | grep -qi "segmentation fault\|segfault"; then
        error_details="${error_details}• SEGMENTATION FAULT detected in logs\n"
    fi

    # get the last few error lines
    local last_errors=$(echo "$logs" | grep -i "error\|exception\|fatal\|panic" | tail -n 5)

    # store analysis in variables for use in email
    CRASH_REASON="$crash_reason"
    CRASH_EXIT_CODE="$exit_code"
    CRASH_ERROR_DETAILS="$error_details"
    CRASH_LAST_ERRORS="$last_errors"
    CRASH_STATE_INFO="$state_info"
}

send_email_alert() {
    local subject="$1"
    local body="$2"

    # using mail
    if command -v mail &> /dev/null; then
        echo "$body" | mail -s "$subject" "$ALERT_EMAIL"
        log_message "Email sent to $ALERT_EMAIL"
    elif command -v sendmail &> /dev/null; then
        echo -e "Subject: $subject\n\n$body" | sendmail "$ALERT_EMAIL"
        log_message "Email sent to $ALERT_EMAIL via sendmail"
    else
        log_message "WARNING: No mail command available for email alerts"
        # Also write to a pending alerts file
        echo "[$(date)] UNABLE TO SEND EMAIL" >> "$LOG_DIR/pending-alerts.log"
        echo "Subject: $subject" >> "$LOG_DIR/pending-alerts.log"
        echo "Body: $body" >> "$LOG_DIR/pending-alerts.log"
        echo "---" >> "$LOG_DIR/pending-alerts.log"
    fi
}

send_webhook_alert() {
    local message="$1"

    if [[ -n "$ALERT_WEBHOOK" ]]; then
        local escaped_message=$(echo "$message" | \
            sed 's/\\/\\\\/g' | \
            sed 's/"/\\"/g' | \
            sed ':a;N;$!ba;s/\n/\\n/g' | \
            sed 's/\t/\\t/g')

        # create JSON payload
        local payload=$(cat <<EOF
{
    "text": "$escaped_message"
}
EOF
)
        # send webhook with better error handling
        local response=$(curl -X POST "$ALERT_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "$payload" \
            --max-time 10 \
            -w "\nHTTP_CODE:%{http_code}" \
            -s 2>&1)

        # extract HTTP code
        local http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)

        # log webhook result
        if [[ "$http_code" == "200" ]]; then
            log_message "Webhook sent successfully to Teams"
        else
            log_message "WARNING: Webhook failed with HTTP code: $http_code"
            log_message "Response: $response"
            # save failed webhook to file for later review
            echo "[$(date)] HTTP $http_code - Failed to send webhook" >> "$LOG_DIR/webhook-errors.log"
            echo "Payload: $payload" >> "$LOG_DIR/webhook-errors.log"
            echo "Response: $response" >> "$LOG_DIR/webhook-errors.log"
            echo "---" >> "$LOG_DIR/webhook-errors.log"
        fi
    fi
}

# alternative version using jq
send_webhook_alert_with_jq() {
    local message="$1"

    if [[ -n "$ALERT_WEBHOOK" ]]; then
        if command -v jq &> /dev/null; then
            local payload=$(jq -n --arg msg "$message" '{text: $msg}')

            local response=$(curl -X POST "$ALERT_WEBHOOK" \
                -H 'Content-Type: application/json' \
                -d "$payload" \
                --max-time 10 \
                -w "\nHTTP_CODE:%{http_code}" \
                -s 2>&1)

            local http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)

            if [[ "$http_code" == "200" ]]; then
                log_message "Webhook sent successfully to Teams (via jq)"
            else
                log_message "WARNING: Webhook failed with HTTP code: $http_code"
                echo "[$(date)] Failed webhook - HTTP $http_code" >> "$LOG_DIR/webhook-errors.log"
            fi
        else
            log_message "WARNING: jq not installed, falling back to basic webhook"
            send_webhook_alert "$message"
        fi
    fi
}

send_alert() {
    local subject="$1"
    local message="$2"

    log_message "ALERT: $subject"
    log_message "$message"

    send_email_alert "$subject" "$message"
    send_webhook_alert "$subject: $message"
}

get_current_status() {
    if [[ -f "$STATUS_FILE" ]]; then
        cat "$STATUS_FILE"
    else
        echo '{"is_down":false,"down_since":"","last_check":""}'
    fi
}

update_status() {
    local is_down=$1
    local down_since=$2
    local last_check=$(date -Iseconds)

    cat > "$STATUS_FILE" <<EOF
{
  "is_down": $is_down,
  "down_since": "$down_since",
  "last_check": "$last_check"
}
EOF
    log_message "Status updated: is_down=$is_down, down_since=$down_since"
}

calculate_downtime() {
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

log_downtime_event() {
    local down_since=$1
    local recovered_at=$(date -Iseconds)
    local downtime_seconds=$(($(date +%s) - down_since))
    local downtime_formatted=$(calculate_downtime $down_since)

    # append to downtime log
    cat >> "$DOWNTIME_LOG" <<EOF
---
Down Since: $(date -d @$down_since '+%Y-%m-%d %H:%M:%S')
Recovered At: $recovered_at
Duration: $downtime_formatted ($downtime_seconds seconds)
---

EOF
    log_message "Downtime event logged: $downtime_formatted"
}

create_crash_report() {
    local down_since=$1
    local container_id=$(get_container_id)
    local crash_time=$(date -d @$down_since '+%Y-%m-%d %H:%M:%S')
    local downtime_seconds=$(($(date +%s) - down_since))

    # analyze the crash
    analyze_crash_cause "$container_id"

    # append to crash report log (human-readable format)
    cat >> "$CRASH_REPORT" <<EOF
================================================================================
CRASH REPORT
================================================================================
Timestamp:       $crash_time
Container Name:  $CONTAINER_NAME
Container ID:    ${container_id:-N/A}
Downtime:        $(calculate_downtime $down_since) ($downtime_seconds seconds)
Recovered At:    $(date '+%Y-%m-%d %H:%M:%S')

CRASH ANALYSIS:
--------------------------------------------------------------------------------
Reason:          $CRASH_REASON
Exit Code:       ${CRASH_EXIT_CODE:-N/A}
State Info:      ${CRASH_STATE_INFO:-N/A}

Error Indicators:
$CRASH_ERROR_DETAILS

Last Error Messages:
$CRASH_LAST_ERRORS

Last 50 Lines of Container Logs:
--------------------------------------------------------------------------------
EOF

    # add container logs if available
    if [[ -n "$container_id" ]]; then
        docker logs --tail 50 "$container_id" 2>&1 >> "$CRASH_REPORT"
    else
        echo "(No logs available - container ID not found)" >> "$CRASH_REPORT"
    fi

    cat >> "$CRASH_REPORT" <<EOF

================================================================================

EOF

    log_message "Crash report created for event at $crash_time"
}

send_detailed_down_alert() {
    local down_since=$1
    local container_id=$(get_container_id)

    # analyze the crash immediately when container goes down
    analyze_crash_cause "$container_id"

    # build detailed email body
    local email_body="ALERT: Container $CONTAINER_NAME has stopped running

Time of Failure: $down_since
Server: $(hostname)

CRASH ANALYSIS:
===============================================================================
Reason: $CRASH_REASON
Exit Code: ${CRASH_EXIT_CODE:-N/A}
State: ${CRASH_STATE_INFO:-N/A}

"

    if [[ -n "$CRASH_ERROR_DETAILS" ]]; then
        email_body="${email_body}Error Indicators Detected:
$CRASH_ERROR_DETAILS
"
    fi

    if [[ -n "$CRASH_LAST_ERRORS" ]]; then
        email_body="${email_body}
Last Error Messages:
-------------------------------------------------------------------------------
$CRASH_LAST_ERRORS
"
    fi

    # add last 20 lines of logs to email
    if [[ -n "$container_id" ]]; then
        local recent_logs=$(docker logs --tail 20 "$container_id" 2>&1)
        email_body="${email_body}

Last 20 Lines of Container Logs:
===============================================================================
$recent_logs

===============================================================================

Full logs available on server at: $LOG_DIR/crash-reports.log
"
    fi

    email_body="${email_body}
Monitoring will continue and send another alert when container recovers.
"

    send_alert "Container DOWN: $CONTAINER_NAME" "$email_body"
}

send_recovery_alert() {
    local down_since=$1
    local downtime=$2

    local email_body="Container $CONTAINER_NAME has RECOVERED

Downtime Duration: $downtime
Down Since: $(date -d @$down_since '+%Y-%m-%d %H:%M:%S')
Recovered At: $(date '+%Y-%m-%d %H:%M:%S')
Server: $(hostname)

A detailed crash report has been generated at:
$LOG_DIR/crash-reports.log

Container is now running normally.
"

    send_alert "Container RECOVERED: $CONTAINER_NAME" "$email_body"
}

# main Monitoring Loop

monitor_container() {
    log_message "Starting container monitoring for: $CONTAINER_NAME"
    log_message "Alert email: $ALERT_EMAIL"
    log_message "Check interval: $CHECK_INTERVAL seconds"
    log_message "Log directory: $LOG_DIR"

    while true; do
        if is_container_running; then
            local status=$(get_current_status)
            local was_down=$(echo "$status" | grep -o '"is_down":[^,]*' | cut -d: -f2 | tr -d ' ')

            if [[ "$was_down" == "true" ]]; then
                # container recovered
                local down_since=$(echo "$status" | grep -o '"down_since":"[^"]*"' | cut -d'"' -f4)

                if [[ -n "$down_since" ]]; then
                    local down_timestamp=$(date -d "$down_since" +%s 2>/dev/null || echo "")

                    if [[ -n "$down_timestamp" ]]; then
                        local downtime=$(calculate_downtime $down_timestamp)

                        send_recovery_alert $down_timestamp "$downtime"
                        log_downtime_event $down_timestamp
                        create_crash_report $down_timestamp
                    fi
                fi

                update_status false ""
                log_message "Container $CONTAINER_NAME is back up"
            fi

            # always update last check time
            update_status false ""

        else
            # container is down
            local status=$(get_current_status)
            local was_down=$(echo "$status" | grep -o '"is_down":[^,]*' | cut -d: -f2 | tr -d ' ')

            if [[ "$was_down" != "true" ]]; then
                # container just went down, send detailed alert immediately
                local down_since=$(date -Iseconds)

                send_detailed_down_alert "$down_since"
                update_status true "$down_since"
                log_message "Container went down at $down_since"
            else
                # container still down, log ongoing downtime every check
                local down_since=$(echo "$status" | grep -o '"down_since":"[^"]*"' | cut -d'"' -f4)

                if [[ -n "$down_since" ]]; then
                    local down_timestamp=$(date -d "$down_since" +%s 2>/dev/null || echo "")

                    if [[ -n "$down_timestamp" ]]; then
                        local downtime=$(calculate_downtime $down_timestamp)
                        log_message "Container still down. Duration: $downtime"
                    fi
                fi
            fi
        fi

        sleep $CHECK_INTERVAL
    done
}

# entry Point

case "${1:-monitor}" in
    monitor)
        monitor_container
        ;;
    status)
        if is_container_running; then
            echo "Container $CONTAINER_NAME is RUNNING"
            get_container_status
        else
            echo "Container $CONTAINER_NAME is DOWN"
            status=$(get_current_status)
            down_since=$(echo "$status" | grep -o '"down_since":"[^"]*"' | cut -d'"' -f4)
            if [[ -n "$down_since" ]]; then
                down_timestamp=$(date -d "$down_since" +%s 2>/dev/null)
                if [[ -n "$down_timestamp" ]]; then
                    echo "Down for: $(calculate_downtime $down_timestamp)"
                fi
            fi
        fi
        ;;
    report)
        echo "=== Alert Log (Last 50 lines) ==="
        if [[ -f "$ALERT_LOG" ]]; then
            tail -n 50 "$ALERT_LOG"
        else
            echo "(No alerts logged yet)"
        fi
        echo ""
        echo "=== Downtime Report ==="
        if [[ -f "$DOWNTIME_LOG" ]] && [[ -s "$DOWNTIME_LOG" ]]; then
            cat "$DOWNTIME_LOG"
        else
            echo "(No downtime events recorded yet)"
        fi
        echo ""
        echo "=== Crash Reports ==="
        if [[ -f "$CRASH_REPORT" ]] && [[ -s "$CRASH_REPORT" ]]; then
            cat "$CRASH_REPORT"
        else
            echo "(No crash reports generated yet)"
        fi
        ;;
    analyze)
        # manual crash analysis
        echo "Analyzing container crash..."
        container_id=$(get_container_id)
        if [[ -n "$container_id" ]]; then
            analyze_crash_cause "$container_id"
            echo "Crash Reason: $CRASH_REASON"
            echo "Exit Code: $CRASH_EXIT_CODE"
            echo "State: $CRASH_STATE_INFO"
            echo ""
            echo "Error Details:"
            echo -e "$CRASH_ERROR_DETAILS"
            echo ""
            echo "Last Errors:"
            echo "$CRASH_LAST_ERRORS"
        else
            echo "Container ID not found"
        fi
        ;;
    test)
        echo "Running monitoring test (will check once and exit)..."
        if is_container_running; then
            echo "✓ Container is running"
        else
            echo "✗ Container is down"
            echo ""
            echo "Running crash analysis..."
            container_id=$(get_container_id)
            if [[ -n "$container_id" ]]; then
                analyze_crash_cause "$container_id"
                echo "Reason: $CRASH_REASON"
                echo "Exit Code: $CRASH_EXIT_CODE"
            fi
        fi
        echo ""
        echo "Log directory: $LOG_DIR"
        echo "Files present:"
        ls -lh "$LOG_DIR"
        ;;
    *)
        echo "Usage: $0 {monitor|status|report|analyze|test}"
        exit 1
        ;;
esac
