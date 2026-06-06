Add-Type -TypeDefinition @"
using System;
using System.Net;
using System.IO;

public class SimpleServer8080 {
    public static void Start(string rootPath, string prefix) {
        HttpListener listener = new HttpListener();
        listener.Prefixes.Add(prefix);
        listener.Start();
        Console.WriteLine("Server started on " + prefix);
        Console.WriteLine("Press Ctrl+C to stop");

        while (listener.IsListening) {
            HttpListenerContext ctx = listener.GetContext();
            HttpListenerRequest req = ctx.Request;
            HttpListenerResponse resp = ctx.Response;

            string url = req.Url.LocalPath;
            if (url == "/") url = "/index.html";

            string filePath = Path.Combine(rootPath, url.TrimStart('/'));

            if (File.Exists(filePath)) {
                string ext = Path.GetExtension(filePath).ToLower();
                string contentType = "application/octet-stream";
                switch (ext) {
                    case ".html": contentType = "text/html; charset=utf-8"; break;
                    case ".js":   contentType = "application/javascript; charset=utf-8"; break;
                    case ".css":  contentType = "text/css; charset=utf-8"; break;
                    case ".png":  contentType = "image/png"; break;
                    case ".jpg":  contentType = "image/jpeg"; break;
                }

                byte[] buffer = File.ReadAllBytes(filePath);
                resp.ContentType = contentType;
                resp.ContentLength64 = buffer.Length;
                resp.Headers.Add("Access-Control-Allow-Origin", "*");
                resp.OutputStream.Write(buffer, 0, buffer.Length);
                resp.OutputStream.Close();
            } else {
                resp.StatusCode = 404;
                resp.Close();
            }
        }
    }
}
"@ -Language CSharp

[SimpleServer8080]::Start('C:\Users\Lenovo\Downloads\123\stark-shapes', 'http://localhost:8080/')
