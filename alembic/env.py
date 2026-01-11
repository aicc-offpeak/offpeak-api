# alembic/env.py
import os
from dotenv import load_dotenv
from logging.config import fileConfig

load_dotenv()

from sqlalchemy import engine_from_config, pool
from alembic import context

from app.db import Base
from app import models  # noqa: F401  (모델 로드)
# models import가 있어야 autogenerate가 테이블을 인식함

# Alembic Config 객체
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 핵심 1: Alembic이 감지할 메타데이터
target_metadata = Base.metadata

# 핵심 2: .env에서 DATABASE_URL 읽기
def get_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing.")
    return url

def run_migrations_offline():
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
