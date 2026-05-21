from httpx import ASGITransport, AsyncClient
from starlette import status

from AiRecAgent.web.application import get_app


async def test_health() -> None:
    """Checks the health endpoint without requiring a database connection."""
    app = get_app()
    url = app.url_path_for("health_check")
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test", timeout=2.0
    ) as client:
        response = await client.get(url)
    assert response.status_code == status.HTTP_200_OK
