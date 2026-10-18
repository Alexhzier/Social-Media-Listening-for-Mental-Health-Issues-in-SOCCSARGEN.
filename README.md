# Power BI Maps Viewer

A beautiful, modern web application for displaying Power BI maps and reports. This website allows you to embed and visualize Power BI content seamlessly.

## Features

- 🗺️ Display Power BI maps and reports
- 🎨 Modern, responsive UI design
- 🔒 Secure token-based authentication
- 📱 Mobile-friendly interface
- ⚡ Easy configuration and setup

## Quick Start

1. **Open the website**
   - Simply open `index.html` in a web browser
   - Or serve it using a local web server

2. **Get your Power BI Embed URL**
   - Go to [Power BI Service](https://app.powerbi.com)
   - Open your report with maps
   - Click **File → Embed report → Website or portal**
   - Copy the **Embed URL**

3. **Generate an Access Token**
   - You'll need to authenticate using Power BI REST API or Azure AD
   - See the "Authentication Options" section below for details

4. **Load your map**
   - Paste the Embed URL in the configuration panel
   - Enter your Access Token
   - Click **Load Power BI Map**

## Authentication Options

### Option 1: Using Azure AD (Recommended for Production)

You'll need to register an app in Azure AD and use it to get access tokens. Here's a basic example:

```javascript
// Example: Get token using Azure AD
// This requires backend implementation for security
```

### Option 2: Using Power BI REST API

1. Register an app in Azure AD
2. Grant Power BI permissions
3. Use OAuth 2.0 flow to get access tokens
4. Pass the token to the embed URL

### Option 3: Using Service Principal (For Development)

For development/testing, you can use a Service Principal:

1. Create a Service Principal in Azure AD
2. Assign it to your Power BI workspace
3. Use client credentials flow to get tokens

## File Structure

```
.
├── index.html      # Main HTML file
├── styles.css      # Styling and design
├── script.js       # Power BI embedding logic
└── README.md       # This file
```

## Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Internet connection (for Power BI SDK CDN)
- Valid Power BI account and workspace access
- Access token with appropriate permissions

## Security Notes

⚠️ **Important**: This is a client-side application. For production use:

1. **Never expose access tokens in client-side code**
   - Implement a backend service to generate tokens
   - Use server-side authentication
   - Implement token refresh logic

2. **Use Row-Level Security (RLS)**
   - Configure RLS in Power BI
   - Pass user context when embedding

3. **HTTPS Required**
   - Always serve over HTTPS in production
   - Power BI requires secure connections

## Troubleshooting

### "Failed to load Power BI report"
- Verify your Embed URL is correct
- Check that your Access Token is valid and not expired
- Ensure you have permissions to view the report

### "Power BI SDK failed to load"
- Check your internet connection
- Verify CDN is accessible
- Try refreshing the page

### Map not displaying correctly
- Check browser console for errors
- Verify report ID is correct
- Ensure the report contains map visualizations

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Edge (latest)
- Safari (latest)

## License

This project is provided as-is for educational and development purposes.

## Resources

- [Power BI Embedded Documentation](https://docs.microsoft.com/en-us/power-bi/developer/embedded/)
- [Power BI JavaScript SDK](https://github.com/Microsoft/PowerBI-JavaScript)
- [Power BI REST API](https://docs.microsoft.com/en-us/rest/api/power-bi/)

## Support

For Power BI specific issues, refer to:
- [Power BI Community](https://community.powerbi.com/)
- [Power BI Documentation](https://docs.microsoft.com/en-us/power-bi/)

---

Made with ❤️ for Power BI visualization
