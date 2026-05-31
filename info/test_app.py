from la_client import LinearAlgebraClient, LinearAlgebraAPIError

API_KEY = "PUT_REAL_API_KEY_HERE"


def main() -> None:
    try:
        with LinearAlgebraClient(api_key=API_KEY) as client:
            print("health:", client.health())
            print("matmul:", client.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
            print("inv:", client.inv([[4, 7], [2, 6]]))
            print("det:", client.det([[4, 7], [2, 6]]))
            print("rank:", client.rank([[1, 2], [2, 4]]))
    except LinearAlgebraAPIError as e:
        print("API error:", e.status_code, e.message, e.payload)
    except Exception as e:
        print("Unexpected error:", str(e))


if __name__ == "__main__":
    main()
