import boto3

def create_buckets():
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    for bucket in ["raw-documents", "processed-documents", "embeddings"]:
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"✅ Bucket '{bucket}' created")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"⏭️  Bucket '{bucket}' already exists")

def create_table():
    dynamodb = boto3.client(
        "dynamodb",
        endpoint_url="http://localhost:8000",
        region_name="eu-central-1",
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )
    try:
        dynamodb.create_table(
            TableName="docdigit",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print("✅ DynamoDB table 'docdigit' created")
    except dynamodb.exceptions.ResourceInUseException:
        print("⏭️  DynamoDB table 'docdigit' already exists")

if __name__ == "__main__":
    create_buckets()
    create_table()
    print("\n🚀 Local infrastructure ready!")
