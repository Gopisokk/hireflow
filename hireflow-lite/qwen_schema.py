"""
qwen_schema.py — Canonical Resume Pydantic Schema for Qwen2.5-3B-Instruct via Ollama
====================================================================================
Defines the target structured JSON schema for candidate extraction from plain text.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    candidate_name: Optional[str] = Field(default=None, description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    github_url: Optional[str] = Field(default=None, description="GitHub profile URL")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")


class Education(BaseModel):
    institution: Optional[str] = Field(default=None, description="University / College / School name")
    degree: Optional[str] = Field(default=None, description="Degree name, e.g., B.Tech, B.S., M.S.")
    field: Optional[str] = Field(default=None, description="Major or field of study, e.g., Computer Science")
    cgpa: Optional[str] = Field(default=None, description="GPA or percentage if present")
    start_date: Optional[str] = Field(default=None, description="Start date/year")
    end_date: Optional[str] = Field(default=None, description="End date/year")


class Experience(BaseModel):
    organization: Optional[str] = Field(default=None, description="Company / Employer / Lab name")
    role: Optional[str] = Field(default=None, description="Job title or role")
    type: str = Field(
        default="employment",
        description="Category: employment | internship | research | open_source | fellowship | volunteer | other"
    )
    description: List[str] = Field(default_factory=list, description="Bullet points describing duties and achievements")
    technologies: List[str] = Field(default_factory=list, description="Tech stack, programming languages, tools used")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date or Present")
    source_lines: Optional[List[int]] = Field(default_factory=list, description="Optional line numbers if present")


class Project(BaseModel):
    project_name: str = Field(description="Name/Title of the project")
    description: str = Field(default="", description="Detailed description or summary of what was built")
    technologies: List[str] = Field(default_factory=list, description="List of technologies, frameworks, and languages used")
    start_date: Optional[str] = Field(default=None, description="Start date if present")
    end_date: Optional[str] = Field(default=None, description="End date if present")
    url: Optional[str] = Field(default=None, description="Project link, demo, or GitHub repository URL")
    source_lines: Optional[List[int]] = Field(default_factory=list, description="Optional line numbers if present")
    confidence: Optional[str] = Field(default="High", description="Extraction confidence: High | Medium | Low")


class Skills(BaseModel):
    programming_languages: List[str] = Field(default_factory=list, description="Programming languages (e.g. Python, C++)")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks (e.g. React, Flask, Spring Boot)")
    libraries: List[str] = Field(default_factory=list, description="Libraries (e.g. NumPy, PyTorch, SentenceTransformers)")
    databases: List[str] = Field(default_factory=list, description="Databases (e.g. PostgreSQL, Redis, MongoDB)")
    cloud: List[str] = Field(default_factory=list, description="Cloud platforms & DevOps (e.g. AWS, Docker, Kubernetes)")
    tools: List[str] = Field(default_factory=list, description="Developer tools (e.g. Git, VS Code, Postman)")
    other: List[str] = Field(default_factory=list, description="Other technical or domain skills")


class CanonicalResume(BaseModel):
    candidate: Candidate = Field(default_factory=Candidate)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: List[Union[Dict[str, Any], str]] = Field(default_factory=list, description="Certificates earned")
    achievements: List[str] = Field(default_factory=list, description="Honors, awards, hackathon wins")
    competitive_programming: List[str] = Field(default_factory=list, description="LeetCode, Codeforces, Kaggle profiles/ratings")
    volunteering: List[str] = Field(default_factory=list, description="Volunteer work or leadership roles")
