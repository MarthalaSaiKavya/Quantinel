from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from config import get_base_url

Number = Union[int, float]
Matrix = List[List[Number]]
Vector = List[Number]


class LinearAlgebraAPIError(Exception):
    """Raised when the remote linear algebra API returns an error response."""

    def __init__(self, status_code: int, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self.message = message
        self.payload = payload or {}
        super().__init__(f"HTTP {status_code}: {message}")


@dataclass(frozen=True)
class ClientConfig:
    api_key: str
    base_url: str
    timeout: float = 30.0


class LinearAlgebraClient:
    """User-facing SDK for the linear algebra API."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = ClientConfig(
            api_key=api_key,
            base_url=(base_url or get_base_url()).rstrip("/"),
            timeout=timeout,
        )
        self.session = session or requests.Session()

    @classmethod
    def from_default(cls, api_key: str, timeout: float = 30.0) -> "LinearAlgebraClient":
        return cls(api_key=api_key, timeout=timeout)

    @classmethod
    def from_base_url(cls, base_url: str, api_key: str, timeout: float = 30.0) -> "LinearAlgebraClient":
        return cls(api_key=api_key, base_url=base_url, timeout=timeout)

    def docs_url(self) -> str:
        return f"{self.config.base_url}/docs"

    def openapi_schema(self) -> Dict[str, Any]:
        return self._get_public("/openapi.json")

    def health(self) -> Dict[str, Any]:
        return self._get_public("/health")

    def service_info(self) -> Dict[str, Any]:
        return self._get_public("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.config.api_key,
        }

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text}

        if not response.ok:
            detail = data.get("detail", response.text)
            raise LinearAlgebraAPIError(
                status_code=response.status_code,
                message=str(detail),
                payload=data if isinstance(data, dict) else {"raw": data},
            )

        if not isinstance(data, dict):
            return {"result": data}
        return data

    def _get_public(self, path: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.config.base_url}{path}",
            timeout=self.config.timeout,
        )
        return self._handle_response(response)

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        response = self.session.post(
            f"{self.config.base_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        data = self._handle_response(response)
        return data.get("result", data)

    @staticmethod
    def _ensure_matrix(name: str, value: Sequence[Sequence[Number]]) -> Matrix:
        if not isinstance(value, Sequence) or not value:
            raise ValueError(f"{name} must be a non-empty 2D sequence")
        out: Matrix = []
        row_length: Optional[int] = None
        for row in value:
            if not isinstance(row, Sequence) or not row:
                raise ValueError(f"{name} must be a non-empty 2D sequence")
            cast_row = [float(v) for v in row]
            if row_length is None:
                row_length = len(cast_row)
            elif len(cast_row) != row_length:
                raise ValueError(f"{name} must be rectangular")
            out.append(cast_row)
        return out

    @staticmethod
    def _ensure_vector(name: str, value: Sequence[Number]) -> Vector:
        if not isinstance(value, Sequence) or not value:
            raise ValueError(f"{name} must be a non-empty sequence")
        return [float(v) for v in value]

    def matmul(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/matmul", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b)})

    def inv(self, a: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/inv", {"a": self._ensure_matrix("a", a)})

    def det(self, a: Sequence[Sequence[Number]]) -> float:
        return float(self._post("/v1/det", {"a": self._ensure_matrix("a", a)}))

    def eig(self, a: Sequence[Sequence[Number]]) -> Dict[str, Any]:
        return self._post("/v1/eig", {"a": self._ensure_matrix("a", a)})

    def svd(self, a: Sequence[Sequence[Number]]) -> Dict[str, Any]:
        return self._post("/v1/svd", {"a": self._ensure_matrix("a", a)})

    def solve(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/solve", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b)})

    def transpose(self, a: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/transpose", {"a": self._ensure_matrix("a", a)})

    def add(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/add", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b)})

    def sub(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/sub", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b)})

    def hadamard(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/hadamard", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b)})

    def trace(self, a: Sequence[Sequence[Number]]) -> float:
        return float(self._post("/v1/trace", {"a": self._ensure_matrix("a", a)}))

    def norm(self, a: Sequence[Sequence[Number]]) -> float:
        return float(self._post("/v1/norm", {"a": self._ensure_matrix("a", a)}))

    def rank(self, a: Sequence[Sequence[Number]]) -> int:
        return int(self._post("/v1/rank", {"a": self._ensure_matrix("a", a)}))

    def qr(self, a: Sequence[Sequence[Number]]) -> Dict[str, Matrix]:
        return self._post("/v1/qr", {"a": self._ensure_matrix("a", a)})

    def cholesky(self, a: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/cholesky", {"a": self._ensure_matrix("a", a)})

    def pinv(self, a: Sequence[Sequence[Number]]) -> Matrix:
        return self._post("/v1/pinv", {"a": self._ensure_matrix("a", a)})

    def reshape(self, a: Sequence[Sequence[Number]], rows: int, cols: int) -> Matrix:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        return self._post("/v1/reshape", {"a": self._ensure_matrix("a", a), "rows": rows, "cols": cols})

    def concat(self, a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]], axis: int = 0) -> Matrix:
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        return self._post("/v1/concat", {"a": self._ensure_matrix("a", a), "b": self._ensure_matrix("b", b), "axis": axis})

    def diag(self, values: Sequence[Number]) -> Matrix:
        return self._post("/v1/diag", {"values": self._ensure_vector("values", values)})

    def scalar_mul(self, a: Sequence[Sequence[Number]], scalar: Number) -> Matrix:
        return self._post("/v1/scalar_mul", {"a": self._ensure_matrix("a", a), "scalar": float(scalar)})

    def matrix_power(self, a: Sequence[Sequence[Number]], power: int) -> Matrix:
        return self._post("/v1/matrix_power", {"a": self._ensure_matrix("a", a), "power": int(power)})

    def portfolio_cov_inverse(self, covariance_matrix: Sequence[Sequence[Number]]) -> Matrix:
        return self.inv(covariance_matrix)

    def quadratic_form_components(
        self,
        weights: Sequence[Sequence[Number]],
        covariance_matrix: Sequence[Sequence[Number]],
    ) -> Matrix:
        return self.matmul(weights, covariance_matrix)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "LinearAlgebraClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
