# Seeded SVG Splotch Generator

A deterministic, seed-driven paint splat generator that creates SVG blob shapes.

## Running the App

Since you have Bun installed, you can run the app with:

```bash
# Install dependencies (if you haven't already)
bun install

# Start the development server
bun run dev
```

The app will be available at `http://localhost:5173` (or the port shown in the terminal).

## Building for Production

```bash
bun run build
```

The built files will be in the `dist` directory.

## Preview Production Build

```bash
bun run preview
```

## Features

- Deterministic generation based on seed strings
- Multiple geometry modes: spray, fling, circle source, line strike
- Real-time preview with density field visualization
- SVG export with customizable size and placement
- Extensive parameter controls for fine-tuning the splat appearance

