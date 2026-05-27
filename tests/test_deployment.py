from pathlib import Path


def test_gpu_dockerfile_uses_cuda_base_image():
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile.gpu"
    assert dockerfile.exists(), "GPU Dockerfile must exist for CUDA-compatible deployment"
    content = dockerfile.read_text(encoding="utf-8")
    assert "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04" in content
    assert "python3.11" in content
    assert "ENTRYPOINT [\"/app/scripts/startup.sh\"]" in content


def test_docker_compose_defines_gpu_runtime_and_persistent_volumes():
    compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
    assert compose.exists(), "Docker Compose configuration file is required"
    content = compose.read_text(encoding="utf-8")
    assert "dockerfile: Dockerfile.gpu" in content
    assert "runtime: nvidia" in content
    assert "device_requests:" in content
    assert "./data:/app/data:rw" in content
    assert "./logs:/app/logs:rw" in content
    assert "./backups:/app/backups:rw" in content
    assert "backup:" in content
    assert "prometheus:" in content
    assert "blackbox-exporter:" in content


def test_startup_scripts_exist_and_are_deterministic():
    root = Path(__file__).resolve().parent.parent
    startup = root / "scripts" / "startup.sh"
    entrypoint = root / "scripts" / "entrypoint.sh"
    backup = root / "scripts" / "backup.sh"

    for path in (startup, entrypoint, backup):
        assert path.exists(), f"Deployment script {path.name} must exist"
        text = path.read_text(encoding="utf-8")
        assert "set -eu" in text

    assert "APP_ENV=\"${APP_ENV:-production}\"" in startup.read_text(encoding="utf-8")
    assert "mkdir -p /app/data /app/logs /app/backups" in entrypoint.read_text(encoding="utf-8")
    assert "tar -czf \"$archive\" -C \"$BACKUP_SOURCE\" ." in backup.read_text(encoding="utf-8")


def test_monitoring_configuration_points_to_blackbox_exporter():
    root = Path(__file__).resolve().parent.parent
    prometheus = root / "monitoring" / "prometheus.yml"
    blackbox = root / "monitoring" / "blackbox.yml"
    assert prometheus.exists()
    assert blackbox.exists()

    prometheus_text = prometheus.read_text(encoding="utf-8")
    blackbox_text = blackbox.read_text(encoding="utf-8")

    assert "job_name: 'blackbox-health'" in prometheus_text
    assert "blackbox-exporter:9115" in prometheus_text
    assert "module: [http_2xx]" in prometheus_text
    assert "prober: http" in blackbox_text
    assert "valid_http_versions: [\"HTTP/1.1\", \"HTTP/2\"]" in blackbox_text
