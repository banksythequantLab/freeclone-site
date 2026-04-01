const BACKEND_URL = 'https://gpu.freeclone.net';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const method = request.method;

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Content-Type': 'application/json'
    };

    // Handle CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // ============= API ROUTES =============

    // POST /api/upload
    if (method === 'POST' && url.pathname === '/api/upload') {
      try {
        const formData = await request.formData();
        const file = formData.get('file');
        const urlParam = formData.get('url');

        if (!file && !urlParam) {
          return new Response(JSON.stringify({ error: 'No file or URL provided' }), {
            status: 400,
            headers: corsHeaders
          });
        }

        // Generate job ID
        const jobId = 'job_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

        // TODO: In production, upload to R2 or backend
        // For now, we'll return a successful stub

        return new Response(JSON.stringify({
          jobId,
          status: 'uploaded',
          message: 'File received. Processing will start shortly.',
          size: file ? file.size : 0,
          backendUrl: BACKEND_URL
        }), {
          status: 200,
          headers: corsHeaders
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 400,
          headers: corsHeaders
        });
      }
    }

    // POST /api/process
    if (method === 'POST' && url.pathname === '/api/process') {
      try {
        const body = await request.json();
        const { jobId, service, sourceLanguage, targetLanguage, hasTranscript, scriptText, captionOptions } = body;

        if (!jobId) {
          return new Response(JSON.stringify({ error: 'jobId required' }), {
            status: 400,
            headers: corsHeaders
          });
        }

        // Forward to backend (if available) or return stub
        try {
          const backendResponse = await fetch(`${BACKEND_URL}/api/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jobId,
              service,
              sourceLanguage,
              targetLanguage,
              hasTranscript,
              scriptText,
              captionOptions
            })
          });

          if (backendResponse.ok) {
            return new Response(backendResponse.body, {
              status: 200,
              headers: corsHeaders
            });
          }
        } catch (backendErr) {
          // Backend not available, return stub
        }

        // Stub response when backend not connected
        return new Response(JSON.stringify({
          jobId,
          status: 'processing',
          message: 'Job queued. Backend service not yet connected.',
          service,
          backendUrl: BACKEND_URL
        }), {
          status: 202,
          headers: corsHeaders
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 400,
          headers: corsHeaders
        });
      }
    }

    // GET /api/job/:id
    if (method === 'GET' && url.pathname.match(/^\/api\/job\/[a-zA-Z0-9_-]+$/)) {
      const jobId = url.pathname.split('/').pop();

      try {
        // Try to get status from backend
        try {
          const backendResponse = await fetch(`${BACKEND_URL}/api/job/${jobId}`);
          if (backendResponse.ok) {
            const data = await backendResponse.json();
            return new Response(JSON.stringify(data), {
              status: 200,
              headers: corsHeaders
            });
          }
        } catch (backendErr) {
          // Backend not available
        }

        // Stub response: simulate completed job after a delay
        const jobNumber = parseInt(jobId.split('_')[1]) || 0;
        const ageMs = Date.now() - jobNumber;

        if (ageMs > 3000) {
          // Pretend it's done after 3 seconds
          return new Response(JSON.stringify({
            jobId,
            status: 'done',
            message: 'Processing complete',
            audioUrl: '/sample-audio.mp3',
            transcript: 'This is a sample transcript. The backend service is not connected yet. Connect a GPU server to localhost:8000 to process real jobs.',
            captions: [
              { start: '00:00:00,000', end: '00:00:05,000', text: 'Sample caption line 1' },
              { start: '00:00:05,000', end: '00:00:10,000', text: 'Sample caption line 2' }
            ]
          }), {
            status: 200,
            headers: corsHeaders
          });
        } else {
          // Still processing
          return new Response(JSON.stringify({
            jobId,
            status: 'processing',
            progress: Math.min(ageMs / 3000 * 100, 99),
            message: 'Processing your request...'
          }), {
            status: 200,
            headers: corsHeaders
          });
        }
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 400,
          headers: corsHeaders
        });
      }
    }

    // POST /api/url-extract
    if (method === 'POST' && url.pathname === '/api/url-extract') {
      try {
        const body = await request.json();
        const { url: videoUrl } = body;

        if (!videoUrl) {
          return new Response(JSON.stringify({ error: 'url parameter required' }), {
            status: 400,
            headers: corsHeaders
          });
        }

        // Try backend first
        try {
          const backendResponse = await fetch(`${BACKEND_URL}/api/url-extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: videoUrl })
          });

          if (backendResponse.ok) {
            const data = await backendResponse.json();
            return new Response(JSON.stringify(data), {
              status: 200,
              headers: corsHeaders
            });
          }
        } catch (backendErr) {
          // Backend not available
        }

        // Stub response
        return new Response(JSON.stringify({
          title: 'Extracted Video',
          audioUrl: '/sample-audio.mp3',
          duration: 120,
          message: 'Backend not connected. Using sample audio.'
        }), {
          status: 200,
          headers: corsHeaders
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 400,
          headers: corsHeaders
        });
      }
    }

    // GET /health
    if (method === 'GET' && url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok',
        worker: 'freeclone-landing',
        domain: url.hostname,
        variant: url.hostname.includes('freevoiceclone') ? 'consumer-seo' : 'b2b-primary',
        timestamp: new Date().toISOString(),
        backendConnected: false,
        backendUrl: BACKEND_URL
      }), { headers: corsHeaders });
    }

    // Default: serve static assets
    return env.ASSETS.fetch(request);
  }
};
