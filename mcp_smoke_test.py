"""
Standalone script (not a pytest test) that proves the MCP server actually
works end-to-end over the real protocol: spawns server.py as a subprocess,
connects via stdio using the MCP client SDK, and calls tools through the
wire protocol -- not just importing Python functions directly.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "mcp_server.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"Connected. Server exposes {len(tools.tools)} tools:")
            for t in tools.tools:
                print(f"  - {t.name}")

            print("\n--- Calling lookup_vin over the wire ---")
            result = await session.call_tool("lookup_vin", {"vin": "1FTFW1ET5DFC10312"})
            print(result.content[0].text)

            print("\n--- Calling check_compliance over the wire ---")
            result = await session.call_tool("check_compliance", {"listing_id": "L-1005"})
            print(result.content[0].text)

            print("\n--- Proposing a price update over the wire ---")
            result = await session.call_tool(
                "update_listing_price",
                {"listing_id": "L-1001", "new_price": 29999, "requested_by": "smoke_test"},
            )
            print(result.content[0].text)

            print("\n--- Calling a nonexistent VIN to confirm error handling over the wire ---")
            result = await session.call_tool("lookup_vin", {"vin": "BOGUS"})
            print("isError:", result.isError)
            print(result.content[0].text)

    print("\nSmoke test complete -- real MCP protocol round-trip confirmed working.")


if __name__ == "__main__":
    asyncio.run(main())
