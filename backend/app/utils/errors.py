"""Domain errors that the API layer translates into clean HTTP responses."""

from __future__ import annotations


class DepthWizardError(Exception):
    """Base class for all expected (non-bug) failures."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(DepthWizardError):
    status_code = 400
    code = "invalid_image"


class PayloadTooLargeError(DepthWizardError):
    status_code = 413
    code = "payload_too_large"


class ModelError(DepthWizardError):
    status_code = 503
    code = "model_error"


class JobNotFoundError(DepthWizardError):
    status_code = 404
    code = "job_not_found"
