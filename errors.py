from fastapi.responses import JSONResponse


class TTSError(Exception):
    def __init__(self, message: str, error_type: str, code: str, status_code: int = 400):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.status_code = status_code


def openai_error_response(message: str, error_type: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )
