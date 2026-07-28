const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status = null, details = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export class ApiClient {
  constructor(getToken, onUnauthorized) {
    this.getToken = getToken;
    this.onUnauthorized = onUnauthorized;
  }

  async request(path, options = {}) {
    const { body, binary = false, timeoutMs = 600_000, ...fetchOptions } = options;
    const token = this.getToken?.();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${API_URL}${path}`, {
        ...fetchOptions,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        headers: {
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...fetchOptions.headers,
        },
      });

      if (response.status === 401) {
        this.onUnauthorized?.();
      }
      if (!response.ok) {
        let payload = null;
        try {
          payload = await response.json();
        } catch {
          // A proxy or development server may return a plain-text error.
        }
        const detail = payload?.detail ?? payload?.message;
        const message =
          Array.isArray(detail)
            ? detail.map((item) => `${item.loc?.slice(-1)?.[0] || "request"}: ${item.msg}`).join("; ")
            : typeof detail === "object"
            ? detail.message || `Request failed (HTTP ${response.status}).`
            : detail || `Request failed (HTTP ${response.status}).`;
        throw new ApiError(message, response.status, typeof detail === "object" && !Array.isArray(detail) ? detail : payload);
      }
      if (response.status === 204) return null;
      return binary ? response.blob() : response.json();
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error.name === "AbortError") {
        throw new ApiError("The request took too long. Check the API terminal before trying again.");
      }
      throw new ApiError("The platform API is not reachable. Start the FastAPI service first.");
    } finally {
      window.clearTimeout(timeout);
    }
  }

  get(path, options) {
    return this.request(path, { ...options, method: "GET" });
  }

  post(path, body, options) {
    return this.request(path, { ...options, method: "POST", body });
  }

  delete(path) {
    return this.request(path, { method: "DELETE" });
  }

  download(path) {
    return this.request(path, { method: "GET", binary: true });
  }
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
