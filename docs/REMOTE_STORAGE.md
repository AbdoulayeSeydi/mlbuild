# Remote Storage Setup

MLBuild works locally out-of-the-box. Remote storage is **optional** for team collaboration and CI/CD.

## Quick Start (Local Only)
```bash
pip install mlbuild
mlbuild build --model model.onnx --target apple_m1
mlbuild validate abc123 --max-p95 100
```

No setup required! Models stored in `.mlbuild/` directory.

---

## Team Collaboration Setup

Need to share models with teammates or CI/CD? Set up remote storage once.

### Option 1: Cloudflare R2 (Recommended - Free 10GB)

**Why R2?**
- ✅ 10GB free forever
- ✅ Zero egress fees (downloads are free)
- ✅ S3-compatible API
- ✅ Fast global CDN

**Setup (5 minutes):**

1. **Create R2 Bucket**
   - Go to [dash.cloudflare.com](https://dash.cloudflare.com)
   - Navigate to R2 Object Storage
   - Click "Create bucket"
   - Name: `mlbuild-models`

2. **Create API Token**
   - Click "Manage R2 API Tokens"
   - Click "Create API Token"
   - Permissions: "Object Read & Write"
   - Copy: Account ID, Access Key ID, Secret Access Key

3. **Configure Credentials**
```bash
   # Add to ~/.bashrc or ~/.zshrc
   export AWS_ACCESS_KEY_ID=your-r2-access-key-id
   export AWS_SECRET_ACCESS_KEY=your-r2-secret-access-key
```

4. **Add Remote to MLBuild**
```bash
   mlbuild remote add production \
     --backend s3 \
     --bucket mlbuild-models \
     --region auto \
     --endpoint https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com \
     --default
```

5. **Push/Pull Models**
```bash
   # Push model to cloud
   mlbuild push abc123
   
   # Pull model on another machine
   mlbuild pull abc123
```

---

### Option 2: AWS S3

**Setup:**

1. **Create S3 Bucket**
```bash
   # Using AWS CLI
   aws s3 mb s3://my-company-mlbuild-models --region us-east-1
```
   
   Or via [AWS Console](https://console.aws.amazon.com/s3)

2. **Create IAM User with S3 Access**
   - Go to IAM → Users → Create user
   - Attach policy: `AmazonS3FullAccess` (or custom policy)
   - Create access key
   - Copy: Access Key ID, Secret Access Key

3. **Configure AWS Credentials**
```bash
   # Option A: AWS CLI
   aws configure
   
   # Option B: Environment variables
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
```

4. **Add Remote to MLBuild**
```bash
   mlbuild remote add production \
     --backend s3 \
     --bucket my-company-mlbuild-models \
     --region us-east-1 \
     --default
```

5. **Push/Pull Models**
```bash
   mlbuild push abc123
   mlbuild pull abc123
```

---

### Option 3: Backblaze B2

**Setup:**

1. Create bucket at [backblaze.com](https://www.backblaze.com/b2/cloud-storage.html)
2. Create application key
3. Configure credentials
4. Add remote (S3-compatible endpoint)
```bash
mlbuild remote add production \
  --backend s3 \
  --bucket mlbuild-models \
  --region us-west-000 \
  --endpoint https://s3.us-west-000.backblazeb2.com
```

---

## CI/CD Integration

### GitHub Actions
```yaml
name: Validate Model Performance

on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install MLBuild
        run: pip install mlbuild
      
      - name: Configure Remote Storage
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          mlbuild remote add prod \
            --backend s3 \
            --bucket company-models \
            --region us-east-1
      
      - name: Pull and Validate Model
        run: |
          mlbuild pull production
          mlbuild validate abc123 --max-p95 100 --ci
```

**Required Secrets:**
- Go to repo Settings → Secrets → Actions
- Add `AWS_ACCESS_KEY_ID`
- Add `AWS_SECRET_ACCESS_KEY`

---

## Managing Multiple Remotes
```bash
# Add multiple remotes
mlbuild remote add dev --backend s3 --bucket dev-models --region us-east-1
mlbuild remote add staging --backend s3 --bucket staging-models --region us-east-1
mlbuild remote add prod --backend s3 --bucket prod-models --region us-east-1 --default

# List remotes
mlbuild remote list

# Push to specific remote
mlbuild push abc123 dev
mlbuild push abc123 staging
mlbuild push abc123 prod

# Set default
mlbuild remote set-default prod
```

---

## Security Best Practices

### 1. Use IAM Roles (AWS)
Instead of access keys, use IAM roles for EC2/ECS/Lambda:
```bash
# No credentials needed - uses instance role
mlbuild remote add prod --backend s3 --bucket models
```

### 2. Restrict Bucket Access
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ],
    "Resource": [
      "arn:aws:s3:::mlbuild-models/*",
      "arn:aws:s3:::mlbuild-models"
    ]
  }]
}
```

### 3. Use Separate Credentials for CI
Create read-only credentials for CI pull operations.

---

## Troubleshooting

### "AWS credentials not found"
```bash
# Verify credentials are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Or check AWS config
cat ~/.aws/credentials
```

### "Bucket does not exist"
```bash
# Verify bucket exists
aws s3 ls s3://your-bucket-name

# Create if missing
aws s3 mb s3://your-bucket-name
```

### "Access denied"
- Check IAM permissions include S3 read/write
- Verify bucket policy allows access
- Check credentials are for correct AWS account

---

## Cost Estimates

### Cloudflare R2
- **Free tier:** 10GB storage, 10M reads/month
- **Paid:** $0.015/GB/month storage, $0/GB egress ✅
- **Example:** 100GB models = $1.50/month

### AWS S3
- **Free tier:** 5GB for 12 months (new accounts)
- **Paid:** $0.023/GB/month storage, $0.09/GB egress
- **Example:** 100GB models + 1TB downloads = $92.30/month

### Backblaze B2
- **Free tier:** 10GB storage, 1GB/day downloads
- **Paid:** $0.006/GB/month storage, $0.01/GB egress
- **Example:** 100GB models + 1TB downloads = $10.60/month

**Recommendation:** Use Cloudflare R2 for zero egress fees.