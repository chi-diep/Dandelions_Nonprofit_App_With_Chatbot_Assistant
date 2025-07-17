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

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

app.use(express.static(path.join(path.resolve(), 'static')));

if (!process.env.SENDGRID_API_KEY) {
  console.error("No SENDGRID_API_KEY found in .env. Exiting...");
  process.exit(1);
}
sgMail.setApiKey(process.env.SENDGRID_API_KEY);
console.log('Loaded SendGrid Key:', process.env.SENDGRID_API_KEY.slice(0, 8));

function readJsonFile(filename) {
  const filepath = path.join('Data- JSON format', filename);
  if (!fs.existsSync(filepath)) return [];
  return JSON.parse(fs.readFileSync(filepath, 'utf8'));
}

let currentCodes = {};

setInterval(() => {
  const now = Date.now();
  for (const [email, record] of Object.entries(currentCodes)) {
    if (now > record.expiresAt) delete currentCodes[email];
  }
}, 60000);

app.post('/api/send-code', async (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: "Email is required" });

  const code = Math.floor(100000 + Math.random() * 900000);
  const expiresAt = Date.now() + 10 * 60 * 1000;
  currentCodes[email] = { code, expiresAt };

  const htmlContent = `
  <div style="max-width:600px;margin:auto;font-family:sans-serif;border-radius:10px;overflow:hidden;border:1px solid #e0e0e0;box-shadow:0 4px 20px rgba(0,0,0,0.05);">
    <div style="background:linear-gradient(135deg, #d946ef, #facc15);padding:25px;text-align:center;">
      <h1 style="color:#fff;font-size:26px;">Welcome to Dandelions!</h1>
      <p style="color:#ecf0f1;font-size:14px;">Let’s keep your account secure & our garden blooming</p>
    </div>
    <div style="padding:30px;background:#fff;">
      <p style="font-size:16px;color:#333;">Hello friend,</p>
      <p style="font-size:16px;color:#333;">Use this code to continue. It's valid for <strong>10 minutes</strong>.</p>
      <div style="text-align:center;margin:30px 0;">
        <div style="display:inline-block;background:#f0f9f5;border:2px dashed #facc15;border-radius:8px;padding:18px 35px;font-size:32px;color:#d946ef;letter-spacing:4px;font-weight:bold;">
          ${code}
        </div>
      </div>
      <p style="font-size:14px;color:#555;">If this wasn’t you, just ignore this email. Your data stays safe.</p>
    </div>
    <div style="background:#f6f6f6;text-align:center;padding:18px;">
      <small style="color:#999;">&copy; ${new Date().getFullYear()} Dandelions. All rights reserved.</small>
    </div>
  </div>`;

  try {
    await sgMail.send({
      to: email,
      from: { email: 'info@dandelions.org', name: 'Dandelions Security Team' },
      subject: 'This is your Dandelions verification code',
      html: htmlContent,
      text: `Hi there,\n\nYour one-time Dandelions code is: ${code}\n\nIt expires in 10 minutes.\n\n— Dandelions Team`
    });
    console.log(`Sent code ${code} to ${email}`);
    res.json({ message: "Code sent" });
  } catch (err) {
    console.error('SendGrid error:', err.response?.body || err);
    res.status(500).json({ error: "Failed to send email" });
  }
});

app.post('/api/verify-code', async (req, res) => {
  const { email, code } = req.body;
  if (!email || !code) return res.status(400).json({ error: "Email and code required" });

  const record = currentCodes[email];
  if (!record) return res.status(401).json({ error: "No code found or already used" });
  if (Date.now() > record.expiresAt) {
    delete currentCodes[email];
    return res.status(401).json({ error: "Code expired" });
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

    console.log(`Verified ${email} as ${role}`);
    return res.json({ verified: true, role });
  }

  res.status(401).json({ error: "Invalid code" });
});

app.get('/api/signups', (req, res) => {
  try {
    const data = readJsonFile('signups_data.json');
    res.json(data);
  } catch (err) {
    console.error('Error reading signups_data.json:', err);
    res.status(500).json({ error: 'Could not fetch signups' });
  }
});

app.post('/api/signup', async (req, res) => {
  const { name, email, phone, title, date, hours } = req.body;
  console.log(`New signup: ${name} for ${title} on ${date}`);

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

    const htmlContent = `
    <div style="max-width:600px;margin:auto;font-family:sans-serif;border-radius:10px;overflow:hidden;border:1px solid #e0e0e0;box-shadow:0 4px 20px rgba(0,0,0,0.05);">
      <div style="background:linear-gradient(135deg,#d946ef,#facc15);padding:25px;text-align:center;">
        <h2 style="color:#fff;font-size:24px;">Thanks for signing up, ${name}!</h2>
        <p style="color:#ecf0f1;font-size:14px;">We are excited to see you at Dandelions!</p>
      </div>
      <div style="padding:30px;background:#fff;">
        <p style="font-size:16px;color:#333;">You’re confirmed for:</p>
        <ul style="font-size:16px;color:#555;">
          <li><strong>Shift:</strong> ${title}</li>
          <li><strong>Date:</strong> ${date}</li>
          <li><strong>Hours:</strong> ${hours}</li>
        </ul>
        <p style="margin-top:20px;font-size:14px;color:#555;">Reply anytime if you have questions. We can’t wait to make memories together.</p>
      </div>
      <div style="background:#f6f6f6;text-align:center;padding:18px;">
        <small style="color:#999;">&copy; ${new Date().getFullYear()} Dandelions. All rights reserved.</small>
      </div>
    </div>`;

    await sgMail.send({
      to: email,
      from: { email: 'cheeyippi@gmail.com', name: 'Dandelions Team' },
      subject: `You're signed up for "${title}" on ${date}`,
      html: htmlContent,
      text: `Hi ${name},\n\nYou’re signed up for "${title}" on ${date} for ${hours} hours.\n\nSee you soon!\n\n— Dandelions Team`
    });

    res.json({ success: true });
  } catch (err) {
    console.error('Signup write or SendGrid error:', err);
    res.status(500).json({ error: 'Could not save signup' });
  }
});

app.get('/api/shifts', (req, res) => {
  try { res.json(readJsonFile('shifts.json')); }
  catch (err) {
    console.error('Error reading shifts.json:', err);
    res.status(500).json({ error: 'Could not fetch shifts' });
  }
});

app.get('/api/kits', (req, res) => {
  try { res.json(readJsonFile('kits.json')); }
  catch (err) {
    console.error('Error reading kits.json:', err);
    res.status(500).json({ error: 'Could not fetch kits' });
  }
});

app.get('/api/volunteers', (req, res) => {
  try { res.json(readJsonFile('volunteers.json')); }
  catch (err) {
    console.error('Error reading volunteers.json:', err);
    res.status(500).json({ error: 'Could not fetch volunteers' });
  }
});

app.get('/api/personal-stories', (req, res) => {
  try { res.json(readJsonFile('personal_stories.json')); }
  catch (err) {
    console.error('Error reading personal_stories.json:', err);
    res.status(500).json({ error: 'Could not fetch personal stories' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

app.use((req, res) => {
  res.status(404).json({ error: "Not Found" });
});
app.post('/api/ask', async (req, res) => {
  const { email, question } = req.body;

  if (!email || !question) {
    return res.status(400).json({ error: "Email and question are required." });
  }

  // Check if user is MFA verified
  const record = currentCodes[email];
  if (!record || Date.now() > record.expiresAt) {
    return res.status(403).json({ error: "Unauthorized or session expired." });
  }

  try {
    const response = await fetch('https://your-flask-backend.onrender.com/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, question })
    });

    const result = await response.json();
    res.json(result);
  } catch (err) {
    console.error("Error calling Flask RAG service:", err);
    res.status(500).json({ error: "Failed to get answer from RAG service." });
  }
});

app.listen(port, () => {
  console.log(` Dandelions server running on http://localhost:${port}`);
});
