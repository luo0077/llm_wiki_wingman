import argparse

import uvicorn

from .app import create_app


def main() -> None:
    p = argparse.ArgumentParser(prog="llm-wiki-server")
    p.add_argument("--host", default=None, help="default: LLM_WIKI_BIND_HOST, else 0.0.0.0 if apiConfig.allowLanAccess, else 127.0.0.1")
    p.add_argument("--port", type=int, default=19828)
    a = p.parse_args()
    from .server_bind import configured_bind_host

    uvicorn.run(create_app(), host=a.host or configured_bind_host(), port=a.port)


if __name__ == "__main__":
    main()
