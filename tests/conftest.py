import os

# Disable testcontainers' Ryuk reaper before any subpackage imports it.
#
# Ryuk is a side-car container that watches the test process and cleans up
# other containers when it exits. On macOS Docker Desktop, rapid sequential
# test sessions can collide on the Ryuk container name and fail with a 409
# Conflict, blocking the whole dialect test matrix. Disabling Ryuk skips the
# coordinator entirely; spawned postgres/mysql containers stop on their own
# when the test using them exits. The only downside is that if a test
# process is killed (SIGKILL, OOM), its dialect containers leak — clean up
# manually with `docker container prune`.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
