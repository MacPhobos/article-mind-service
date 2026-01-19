# CORS Configuration Guide

## Overview

This service supports flexible CORS configuration for development and production environments.

## Configuration Options

### Option 1: Specific Origins (Production/Recommended)

Set `CORS_ORIGINS` to a comma-separated list of allowed origins:

```env
CORS_ORIGINS=http://localhost:5173,http://localhost:5174,http://192.168.1.9:5175
CORS_ALLOW_ALL=false
```

**When to use:**
- Production deployments
- When you know the exact frontend origins
- For maximum security

### Option 2: Allow All Origins (Development Only)

Set `CORS_ALLOW_ALL=true` to allow requests from ANY origin:

```env
CORS_ALLOW_ALL=true
```

**When to use:**
- Local development with changing IP addresses
- Testing from mobile devices on local network
- Quick prototyping

**⚠️ WARNING:** NEVER use `CORS_ALLOW_ALL=true` in production! This disables CORS protection entirely.

## How It Works

The CORS middleware configuration in `main.py` checks the settings:

1. If `CORS_ALLOW_ALL=true`: Uses `allow_origins=["*"]` (all origins allowed)
2. If `CORS_ALLOW_ALL=false`: Uses the origins from `CORS_ORIGINS` environment variable

## Implementation Details

### Files Modified

1. **src/article_mind_service/config.py**
   - Added `cors_allow_all: bool = False` setting
   - Updated `cors_origins_list` property to return `["*"]` when `cors_allow_all=True`

2. **src/article_mind_service/main.py**
   - Added comment explaining CORS behavior
   - No code changes needed (uses `settings.cors_origins_list`)

3. **.env.example**
   - Documented `CORS_ALLOW_ALL` setting with security warning

## Testing

To test the CORS configuration:

```bash
# Start the service with CORS_ALLOW_ALL enabled
echo "CORS_ALLOW_ALL=true" >> .env
make dev

# From frontend (any origin will work):
curl -H "Origin: http://any-origin.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS http://localhost:8000/api/v1/sessions -v

# Check for Access-Control-Allow-Origin: * in response headers
```

## Troubleshooting

### Issue: Still getting CORS errors after enabling CORS_ALLOW_ALL

**Solution:**
1. Verify `.env` file contains `CORS_ALLOW_ALL=true`
2. Restart the development server (`make dev`)
3. Check server logs for "Starting Article Mind Service" message

### Issue: CORS works locally but not in production

**Solution:**
- NEVER use `CORS_ALLOW_ALL=true` in production
- Add your production frontend domain to `CORS_ORIGINS`
- Example: `CORS_ORIGINS=https://app.example.com,https://www.example.com`

### Issue: Preflight (OPTIONS) requests failing

**Solution:**
- Ensure `allow_methods=["*"]` and `allow_headers=["*"]` in CORS middleware (already configured)
- Check that the service is receiving the OPTIONS request (check logs)
- Verify CORS configuration took effect (restart server)

## Security Best Practices

1. **Development:** Use `CORS_ALLOW_ALL=true` for convenience
2. **Staging:** Use specific origins matching your staging frontend URL
3. **Production:** Use specific origins matching your production frontend URL(s)
4. **Never:** Commit `.env` with `CORS_ALLOW_ALL=true` to version control

## Example Configurations

### Local Development (Same Machine)
```env
CORS_ORIGINS=http://localhost:5173,http://localhost:5174
CORS_ALLOW_ALL=false
```

### Local Development (Network Testing)
```env
CORS_ALLOW_ALL=true
```

### Production
```env
CORS_ORIGINS=https://article-mind.example.com,https://www.article-mind.example.com
CORS_ALLOW_ALL=false
```

## References

- FastAPI CORS Middleware: https://fastapi.tiangolo.com/tutorial/cors/
- MDN CORS Documentation: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
