# Transformer Architecture Visualizer

Interactive visualization tool for understanding transformer neural network architecture. Built with React and Express.

## What This Does

Visualizes the internal workings of a transformer model when processing text input. Shows the progression through encoder/decoder layers, attention mechanisms, and embedding transformations.

## Tech Stack

- Frontend: React, TypeScript, D3.js, Framer Motion
- Backend: Node.js, Express, TypeScript
- Deployment: AWS EC2 (see DEPLOYMENT.md)

## Development Setup

### Prerequisites

- Node.js 16 or higher
- npm

### Installation

```bash
# Install root dependencies
npm install

# Install client dependencies
cd client
npm install
cd ..
```

### Running Locally

Start both server and client in development mode:

```bash
npm run dev
```

This starts:

- Backend server on http://localhost:5000
- React dev server on http://localhost:3000

Or run them separately:

```bash
# Terminal 1 - Backend
npm run server

# Terminal 2 - Frontend
npm run client
```

## Project Structure

```
transformer-viz/
├── server/
│   ├── index.ts              # Express server
│   ├── types/
│   │   └── transformer.ts    # Type definitions
│   └── utils/
│       └── transformer.ts    # Processing logic
├── client/
│   ├── src/
│   │   ├── App.tsx          # Main app component
│   │   ├── components/      # React components
│   │   ├── api/             # API client
│   │   └── types.ts         # Frontend types
    └── public/
```

## API Endpoints

### POST /api/visualize

Processes text input and returns transformer visualization state.

**Request:**

```json
{
  "text": "Hello world",
  "numLayers": 6,
  "numHeads": 8,
  "embeddingDim": 512
}
```

**Response:**

```json
{
  "state": { /* visualization data */ },
  "metadata": { /* configuration */ }
}
```

## Building for Production

```bash
# Build client
npm run build

# Build server
npm run build:server

# Start production server
npm start
```

## License

MIT
