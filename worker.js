export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      return new Response(JSON.stringify({
        error: 'API not yet connected. Coming soon.',
        docs: 'https://freeclone.net/api-docs'
      }), { status: 503, headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' } });
    }
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok', worker: 'freeclone-landing', domain: url.hostname,
        variant: url.hostname.includes('freevoiceclone') ? 'consumer-seo' : 'b2b-primary',
        timestamp: new Date().toISOString()
      }), { headers: { 'Content-Type': 'application/json' } });
    }
    return env.ASSETS.fetch(request);
  }
};
