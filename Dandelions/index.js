import fetch from 'node-fetch'; 
import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import sgMail from '@sendgrid/mail';
import path from 'path';
import fs from 'fs';

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// Serve static files (frontend)
app.use(express.static(path.join(path.resolve(), 'static')));

// Verify SendGrid key
if (!process.env.SENDGRID_API_KEY) {
  console.error("No SENDGRID_API_KEY found in .env. Exiting...");
  process.exit(1);
}
sgMail.setApiKey(process.env.SENDGRID_API_KEY);
console.log('✅ SendGrid Key loaded');

// Helper: Read JSON file
function readJsonFile(filename) {
  const filepath = path.join('Data- JSON format', filename);
  if (!fs.existsSync(filepath)) return [];
  return JSON.parse(fs.readFileSync(filepath, 'utf8'));
}

// MFA Code Store
let currentCodes = {};
setInterval(() => {
  const now = Date.now();
  for (const [email, record] of Object.entries(currentCodes)) {
    if (now > record.expiresAt) delete currentCodes[email];
  }
}, 60000);

// Send verification code
app.post('/api/send-code', async (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: "Email is required" });

  const code = Math.floor(100000 + Math.random() * 900000);
  const expiresAt = Date.now() + 10 * 60 * 1000;
  currentCodes[email] = { code, expiresAt };

  const htmlContent = `
  <div style="...">Your beautifully styled code email here with code: <strong>${code}</strong></div>
  `;

  try {
    await sgMail.send({
      to: email,
      from: { email: 'info@dandelions.org', name: 'Dandelions Security Team' },
      subject: 'Your Dandelions verification code',
      html: htmlContent,
      text: `Your one-time code is ${code}. It expires in 10 minutes.`
    });

    console.log(`✅ Sent code ${code} to ${email}`);
    res.json({ message: "Code sent" });
  } catch (err) {
    console.error('SendGrid error:', err.response?.body || err);
    res.status(500).json({ error: "Failed to send email" });
  }
});

// Verify code
app.post('/api/verify-code', (req, res) => {
  const { email, code } = req.body;
  if (!email || !code) return res.status(400).json({ error: "Email and code required" });

  const record = currentCodes[email];
  if (!record || Date.now() > record.expiresAt) {
    delete currentCodes[email];
    return res.status(401).json({ error: "Code expired or invalid" });
  }

  if (parseInt(code) === record.code) {
    delete currentCodes[email];
    let role = "volunteer";

    try {
      const data = readJsonFile('volunteers.json');
      const found = data.find(v => v.email === email);
      if (found?.role) role = found.role;
    } catch (err) {
      console.error("Role lookup failed:", err);
    }

    console.log(`✅ Verified ${email} as ${role}`);
    return res.json({ verified: true, role });
  }

  res.status(401).json({ error: "Invalid code" });
});

// Signup handler
app.post('/api/signup', async (req, res) => {
  const { name, email, phone, title, date, hours } = req.body;
  console.log(`📋 New signup: ${name} for ${title} on ${date}`);

  const signup = {
    id: Date.now(),
    name,
    email,
    phone,
    shift_title: title,
    shift_date: date,
    shift_hours: hours,
    created_at: new Date().toISOString()
  };

  try {
    const filepath = path.join('Data- JSON format', 'signups_data.json');
    let data = [];
    if (fs.existsSync(filepath)) {
      data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
    }
    data.push(signup);
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));

    await sgMail.send({
      to: email,
      from: { email: 'cheeyippi@gmail.com', name: 'Dandelions Team' },
      subject: `You're signed up for "${title}" on ${date}`,
      html: `<div style="...">Thanks for signing up!</div>`,
      text: `Hi ${name}, you're confirmed for "${title}" on ${date} for ${hours} hours.`
    });

    res.json({ success: true });
  } catch (err) {
    console.error('Signup error:', err);
    res.status(500).json({ error: 'Could not save signup' });
  }
});

// API endpoints for JSON data
app.get('/api/signups', (req, res) => {
  try { res.json(readJsonFile('signups_data.json')); }
  catch (err) { res.status(500).json({ error: 'Could not fetch signups' }); }
});

app.get('/api/shifts', (req, res) => {
  try { res.json(readJsonFile('shifts.json')); }
  catch (err) { res.status(500).json({ error: 'Could not fetch shifts' }); }
});

app.get('/api/kits', (req, res) => {
  try { res.json(readJsonFile('kits.json')); }
  catch (err) { res.status(500).json({ error: 'Could not fetch kits' }); }
});

app.get('/api/volunteers', (req, res) => {
  try { res.json(readJsonFile('volunteers.json')); }
  catch (err) { res.status(500).json({ error: 'Could not fetch volunteers' }); }
});

app.get('/api/personal-stories', (req, res) => {
  try { res.json(readJsonFile('personal_stories.json')); }
  catch (err) { res.status(500).json({ error: 'Could not fetch personal stories' }); }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

// 404 fallback
app.use((req, res) => {
  res.status(404).json({ error: "Not Found" });
});

app.listen(port, () => {
  console.log(` Dandelions Express server running at http://localhost:${port}`);
});
