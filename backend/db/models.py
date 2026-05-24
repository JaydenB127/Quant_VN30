# -*- coding: utf-8 -*-
"""
SQLAlchemy 2.0 Declarative ORM models representing the complete ETS database schema.
Supports async PostgreSQL operations, robust indexes, and cascade deletes.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import BigInteger, Column, Double, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Cross-dialect JSON type: uses PostgreSQL JSONB for performance/indexing, falls back to standard JSON/TEXT on SQLite
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Declarative Base Class with standard JSON type mapping."""
    type_annotation_map = {
        dict: JSON_TYPE,
        list: JSON_TYPE,
    }


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    storage_backend: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    file_format: Mapped[Optional[str]] = mapped_column(String(20))
    n_rows: Mapped[Optional[int]] = mapped_column(Integer)
    n_columns: Mapped[Optional[int]] = mapped_column(Integer)
    dataset_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    schema_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    profile_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    suggested_target: Mapped[Optional[str]] = mapped_column(String(255))
    suggested_problem_type: Mapped[Optional[str]] = mapped_column(String(50))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    experiments: Mapped[List[Experiment]] = relationship("Experiment", back_populates="dataset")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    dataset_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    pipeline_type: Mapped[str] = mapped_column(String(100), nullable=False)
    config_json: Mapped[Dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSON_TYPE, default=list)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    dataset: Mapped[Optional[Dataset]] = relationship("Dataset", back_populates="experiments")
    runs: Mapped[List[Run]] = relationship("Run", back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="created")  # created, queued, running, completed, failed
    config_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    environment_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column()
    completed_at: Mapped[Optional[datetime]] = mapped_column()
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="runs")
    parameters: Mapped[List[Parameter]] = relationship("Parameter", back_populates="run", cascade="all, delete-orphan")
    metrics: Mapped[List[Metric]] = relationship("Metric", back_populates="run", cascade="all, delete-orphan")
    pipeline_steps: Mapped[List[PipelineStep]] = relationship("PipelineStep", back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[List[Artifact]] = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")


class Parameter(Base):
    __tablename__ = "parameters"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[Run] = relationship("Run", back_populates="parameters")


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Double, nullable=False)
    step: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    run: Mapped[Run] = relationship("Run", back_populates="metrics")


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # pending, running, completed, failed
    outputs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    completed_at: Mapped[Optional[datetime]] = mapped_column()

    run: Mapped[Run] = relationship("Run", back_populates="pipeline_steps")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    artifact_type: Mapped[Optional[str]] = mapped_column(String(50))  # model, plot, report, dataset
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    run: Mapped[Run] = relationship("Run", back_populates="artifacts")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    salt: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
