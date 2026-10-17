# Database Connection Incidents

## Incident: Database Connection Timeout

**Symptoms:** Applications experience intermittent database connection timeouts, failed queries, and increased response latency.

**Possible causes:**
- Database server overload or unavailability.
- Incorrect database connection settings.
- Network connectivity problems between the application and database.
- Connection pool exhaustion.

**Diagnostic checks:**
- Check database server health and availability.
- Review application and database logs.
- Verify connection strings and credentials without exposing secrets.
- Inspect connection pool usage and database connection limits.
- Test network connectivity between the application and database.

**Remediation:**
- Restore database service availability if the server is down.
- Correct invalid connection settings.
- Resolve network or firewall issues.
- Tune connection pool settings after verifying resource limits.

**Important:** These are possible causes and recommended checks, not confirmed root causes.