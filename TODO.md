curl -X POST 'https://api.cloudflare.com/client/v4/accounts/${CF_ID}/browser-rendering/screenshot' \
  -H 'Authorization: Bearer ${CF_API_KEY}' \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://pantip.com/topic/43711908",
    "addStyleTag": [
      {
        "content": "div[id^=\"sp_message_container\"], div[role=\"dialog\"] { display: none !important; } .display-post-wrapper.main-post { position: absolute !important; z-index: 2147483647 !important; width: 480px !important; min-width: 480px !important; overflow: hidden !important; left: 0;  }"
      }
    ],
    "selector": ".display-post-wrapper.main-post",
    "gotoOptions": {
      "waitUntil": "networkidle2",
      "timeout": 45000
    }
  }' \
  --output "test.png"
