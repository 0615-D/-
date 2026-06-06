// --- PATTERN FUNCTIONS ---
function createGrid(i, count) {
    const sideLength = Math.ceil(Math.cbrt(count));
    const spacing = 60 / sideLength;
    const halfGrid = (sideLength - 1) * spacing / 2;
    
    // Determine which side of the cube this particle should be on
    const totalSides = 6; // A cube has 6 sides
    const pointsPerSide = Math.floor(count / totalSides);
    const side = Math.floor(i / pointsPerSide);
    const indexOnSide = i % pointsPerSide;
    
    // Calculate a grid position on a 2D plane
    const sideLength2D = Math.ceil(Math.sqrt(pointsPerSide));
    const ix = indexOnSide % sideLength2D;
    const iy = Math.floor(indexOnSide / sideLength2D);
    
    // Map to relative coordinates (0 to 1)
    const rx = ix / (sideLength2D - 1 || 1);
    const ry = iy / (sideLength2D - 1 || 1);
    
    // Convert to actual coordinates with proper spacing (-halfGrid to +halfGrid)
    const x = rx * spacing * (sideLength - 1) - halfGrid;
    const y = ry * spacing * (sideLength - 1) - halfGrid;
    
    // Place on the appropriate face of the cube
    switch(side % totalSides) {
        case 0: return new THREE.Vector3(x, y, halfGrid); // Front face
        case 1: return new THREE.Vector3(x, y, -halfGrid); // Back face
        case 2: return new THREE.Vector3(x, halfGrid, y); // Top face
        case 3: return new THREE.Vector3(x, -halfGrid, y); // Bottom face
        case 4: return new THREE.Vector3(halfGrid, x, y); // Right face
        case 5: return new THREE.Vector3(-halfGrid, x, y); // Left face
        default: return new THREE.Vector3(0, 0, 0);
    }
}

function createSphere(i, count) {
    // Sphere distribution using spherical coordinates for surface only
    const t = i / count;
    const phi = Math.acos(2 * t - 1); // Full range from 0 to PI
    const theta = 2 * Math.PI * (i / count) * Math.sqrt(count); // Golden ratio distribution
    
    // Fixed radius for surface-only distribution
    const radius = 30;
    
    return new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta) * radius,
        Math.sin(phi) * Math.sin(theta) * radius,
        Math.cos(phi) * radius
    );
}

function createSpiral(i, count) {
    const t = i / count;
    const numArms = 3;
    const armIndex = i % numArms;
    const angleOffset = (2 * Math.PI / numArms) * armIndex;
    const angle = Math.pow(t, 0.7) * 15 + angleOffset;
    const radius = t * 40;
    
    // This is a 2D shape with particles on a thin plane by design 
    const height = 0; // Set to zero or a very small noise value for thickness
    
    return new THREE.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        height
    );
}

function createHelix(i, count) {
    const numHelices = 2;
    const helixIndex = i % numHelices;
    const t = Math.floor(i / numHelices) / Math.floor(count / numHelices);
    const angle = t * Math.PI * 10;
    
    // Fixed radius for surface-only distribution
    const radius = 15;
    const height = (t - 0.5) * 60;
    const angleOffset = helixIndex * Math.PI;
    
    return new THREE.Vector3(
        Math.cos(angle + angleOffset) * radius,
        Math.sin(angle + angleOffset) * radius,
        height
    );
}

function createTorus(i, count) {
    // Torus parameters
    const R = 30; // Major radius (distance from center of tube to center of torus)
    const r = 10; // Minor radius (radius of the tube)
    
    // Use a uniform distribution on the torus surface
    // by using uniform sampling in the 2 angle parameters
    const u = (i / count) * 2 * Math.PI; // Angle around the center of the torus
    const v = (i * Math.sqrt(5)) * 2 * Math.PI; // Angle around the tube
    
    // Parametric equation of a torus
    return new THREE.Vector3(
        (R + r * Math.cos(v)) * Math.cos(u),
        (R + r * Math.cos(v)) * Math.sin(u),
        r * Math.sin(v)
    );
}

function createVortex(i, count) {
    // Vortex parameters
    const height = 60;        // Total height of the vortex
    const maxRadius = 35;     // Maximum radius at the top
    const minRadius = 5;      // Minimum radius at the bottom
    const numRotations = 3;   // Number of full rotations from top to bottom
    
    // Calculate normalized height position (0 = bottom, 1 = top)
    const t = i / count;
    
    // Add some randomness to distribute particles more naturally
    const randomOffset = 0.05 * Math.random();
    const heightPosition = t + randomOffset;
    
    // Calculate radius that decreases from top to bottom
    const radius = minRadius + (maxRadius - minRadius) * heightPosition;
    
    // Calculate angle with more rotations at the bottom
    const angle = numRotations * Math.PI * 2 * (1 - heightPosition) + (i * 0.1);
    
    // Calculate the vertical position (from bottom to top)
    const y = (heightPosition - 0.5) * height;
    
    return new THREE.Vector3(
        Math.cos(angle) * radius,
        y,
        Math.sin(angle) * radius
    );
}

function createGalaxy(i, count) {
    // Galaxy parameters
    const numArms = 4;            // Number of spiral arms
    const armWidth = 0.15;        // Width of each arm (0-1)
    const maxRadius = 40;         // Maximum radius of the galaxy
    const thickness = 5;          // Vertical thickness
    const twistFactor = 2.5;      // How much the arms twist
    
    // Determine which arm this particle belongs to
    const armIndex = i % numArms;
    const indexInArm = Math.floor(i / numArms) / Math.floor(count / numArms);
    
    // Calculate radial distance from center
    const radialDistance = indexInArm * maxRadius;
    
    // Add some randomness for arm width
    const randomOffset = (Math.random() * 2 - 1) * armWidth;
    
    // Calculate angle with twist that increases with distance
    const armOffset = (2 * Math.PI / numArms) * armIndex;
    const twistAmount = twistFactor * indexInArm;
    const angle = armOffset + twistAmount + randomOffset;
    
    // Add height variation that decreases with distance from center
    const verticalPosition = (Math.random() * 2 - 1) * thickness * (1 - indexInArm * 0.8);
    
    return new THREE.Vector3(
        Math.cos(angle) * radialDistance,
        verticalPosition,
        Math.sin(angle) * radialDistance
    );
}

function createWave(i, count) {
    // Wave/ocean parameters
    const width = 60;       // Total width of the wave field
    const depth = 60;       // Total depth of the wave field
    const waveHeight = 10;  // Maximum height of waves
    const waveDensity = 0.1; // Controls wave frequency
    
    // Create a grid of points (similar to your grid function but for a 2D plane)
    const gridSize = Math.ceil(Math.sqrt(count));
    const spacingX = width / gridSize;
    const spacingZ = depth / gridSize;
    
    // Calculate 2D grid position
    const ix = i % gridSize;
    const iz = Math.floor(i / gridSize);
    
    // Convert to actual coordinates with proper spacing
    const halfWidth = width / 2;
    const halfDepth = depth / 2;
    const x = ix * spacingX - halfWidth;
    const z = iz * spacingZ - halfDepth;
    
    // Create wave pattern using multiple sine waves for a more natural look
    // We use the x and z coordinates to create a position-based wave pattern
    const y = Math.sin(x * waveDensity) * Math.cos(z * waveDensity) * waveHeight +
              Math.sin(x * waveDensity * 2.5) * Math.cos(z * waveDensity * 2.1) * (waveHeight * 0.3);
    
    return new THREE.Vector3(x, y, z);
}

function createMobius(i, count) {
    // Möbius strip parameters
    const radius = 25;       // Major radius of the strip
    const width = 10;        // Width of the strip
    
    // Distribute points evenly along the length of the Möbius strip
    // and across its width
    const lengthSteps = Math.sqrt(count);
    const widthSteps = count / lengthSteps;
    
    // Calculate position along length and width of strip
    const lengthIndex = i % lengthSteps;
    const widthIndex = Math.floor(i / lengthSteps) % widthSteps;
    
    // Normalize to 0-1 range
    const u = lengthIndex / lengthSteps;        // Position around the strip (0 to 1)
    const v = (widthIndex / widthSteps) - 0.5;  // Position across width (-0.5 to 0.5)
    
    // Parametric equations for Möbius strip
    const theta = u * Math.PI * 2;  // Full loop around
    
    // Calculate the Möbius strip coordinates
    // This creates a half-twist in the strip
    const x = (radius + width * v * Math.cos(theta / 2)) * Math.cos(theta);
    const y = (radius + width * v * Math.cos(theta / 2)) * Math.sin(theta);
    const z = width * v * Math.sin(theta / 2);
    
    return new THREE.Vector3(x, y, z);
}

function createSupernova(i, count) {
    // Supernova parameters
    const maxRadius = 40;        // Maximum explosion radius
    const coreSize = 0.2;        // Size of the dense core (0-1)
    const outerDensity = 0.7;    // Density of particles in outer shell
    
    // Use golden ratio distribution for even spherical coverage
    const phi = Math.acos(1 - 2 * (i / count));
    const theta = Math.PI * 2 * i * (1 + Math.sqrt(5));
    
    // Calculate radial distance with more particles near center and at outer shell
    let normalizedRadius;
    const random = Math.random();
    
    if (i < count * coreSize) {
        // Dense core - distribute within inner radius
        normalizedRadius = Math.pow(random, 0.5) * 0.3;
    } else {
        // Explosion wave - distribute with more particles at the outer shell
        normalizedRadius = 0.3 + Math.pow(random, outerDensity) * 0.7;
    }
    
    // Scale to max radius
    const radius = normalizedRadius * maxRadius;
    
    // Convert spherical to Cartesian coordinates
    return new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta) * radius,
        Math.sin(phi) * Math.sin(theta) * radius,
        Math.cos(phi) * radius
    );
}

function createKleinBottle(i, count) {
    // Klein Bottle parameters
    const a = 15;          // Main radius
    const b = 4;           // Tube radius
    const scale = 2.5;     // Overall scale
    
    // Use uniform distribution across the surface
    const lengthSteps = Math.ceil(Math.sqrt(count * 0.5));
    const circSteps = Math.ceil(count / lengthSteps);
    
    // Calculate position in the parametric space
    const lengthIndex = i % lengthSteps;
    const circIndex = Math.floor(i / lengthSteps) % circSteps;
    
    // Normalize to appropriate ranges
    const u = (lengthIndex / lengthSteps) * Math.PI * 2;  // 0 to 2π
    const v = (circIndex / circSteps) * Math.PI * 2;      // 0 to 2π
    
    // Klein Bottle parametric equation
    let x, y, z;
    
    // The Klein Bottle has different regions with different parametric equations
    if (u < Math.PI) {
        // First half (handle and transition region)
        x = scale * (a * (1 - Math.cos(u) / 2) * Math.cos(v) - b * Math.sin(u) / 2);
        y = scale * (a * (1 - Math.cos(u) / 2) * Math.sin(v));
        z = scale * (a * Math.sin(u) / 2 + b * Math.sin(u) * Math.cos(v));
    } else {
        // Second half (main bottle body)
        x = scale * (a * (1 + Math.cos(u) / 2) * Math.cos(v) + b * Math.sin(u) / 2);
        y = scale * (a * (1 + Math.cos(u) / 2) * Math.sin(v));
        z = scale * (-a * Math.sin(u) / 2 + b * Math.sin(u) * Math.cos(v));
    }
    
    return new THREE.Vector3(x, y, z);
}

function createFlower(i, count) {
    // Flower/Dandelion parameters
    const numPetals = 12;          // Number of petals
    const petalLength = 25;        // Length of petals
    const centerRadius = 10;       // Radius of center sphere
    const petalWidth = 0.3;        // Width of petals (0-1)
    const petalCurve = 0.6;        // How much petals curve outward (0-1)
    
    // Calculate whether this particle is in the center or on a petal
    const centerParticleCount = Math.floor(count * 0.3); // 30% of particles in center
    const isCenter = i < centerParticleCount;
    
    if (isCenter) {
        // Center particles form a sphere
        const t = i / centerParticleCount;
        const phi = Math.acos(2 * t - 1);
        const theta = 2 * Math.PI * i * (1 + Math.sqrt(5)); // Golden ratio distribution
        
        // Create a sphere for the center
        return new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta) * centerRadius,
            Math.sin(phi) * Math.sin(theta) * centerRadius,
            Math.cos(phi) * centerRadius
        );
    } else {
        // Petal particles
        const petalParticleCount = count - centerParticleCount;
        const petalIndex = i - centerParticleCount;
        
        // Determine which petal this particle belongs to
        const petalId = petalIndex % numPetals;
        const positionInPetal = Math.floor(petalIndex / numPetals) / Math.floor(petalParticleCount / numPetals);
        
        // Calculate angle of this petal
        const petalAngle = (petalId / numPetals) * Math.PI * 2;
        
        // Calculate radial distance from center
        // Use a curve so particles are denser at tip and base
        const radialT = Math.pow(positionInPetal, 0.7); // Adjust density along petal
        const radialDist = centerRadius + (petalLength * radialT);
        
        // Calculate width displacement (thicker at base, thinner at tip)
        const widthFactor = petalWidth * (1 - radialT * 0.7);
        const randomWidth = (Math.random() * 2 - 1) * widthFactor * petalLength;
        
        // Calculate curve displacement (petals curve outward)
        const curveFactor = petalCurve * Math.sin(positionInPetal * Math.PI);
        
        // Convert to Cartesian coordinates
        // Main direction follows the petal angle
        const x = Math.cos(petalAngle) * radialDist + 
                 Math.cos(petalAngle + Math.PI/2) * randomWidth;
        
        const y = Math.sin(petalAngle) * radialDist + 
                 Math.sin(petalAngle + Math.PI/2) * randomWidth;
        
        // Z coordinate creates the upward curve of petals
        const z = curveFactor * petalLength * (1 - Math.cos(positionInPetal * Math.PI));
        
        return new THREE.Vector3(x, y, z);
    }
}

function createFractalTree(i, count) {
    // Fractal Tree parameters
    const trunkLength = 35;        // Initial trunk length
    const branchRatio = 0.67;      // Each branch is this ratio of parent length
    const maxDepth = 6;            // Maximum branching depth
    const branchAngle = Math.PI / 5; // Angle between branches (36 degrees)
    
    // Pre-calculate the total particles needed per depth level
    // Distribute particles more towards deeper levels
    const particlesPerLevel = [];
    let totalWeight = 0;
    
    for (let depth = 0; depth <= maxDepth; depth++) {
        // More branches at deeper levels, distribute particles accordingly
        // Each level has 2^depth branches
        const branches = Math.pow(2, depth);
        const weight = branches * Math.pow(branchRatio, depth);
        totalWeight += weight;
        particlesPerLevel.push(weight);
    }
    
    // Normalize to get actual count per level
    let cumulativeCount = 0;
    const particleCount = [];
    
    for (let depth = 0; depth <= maxDepth; depth++) {
        const levelCount = Math.floor((particlesPerLevel[depth] / totalWeight) * count);
        particleCount.push(levelCount);
        cumulativeCount += levelCount;
    }
    
    // Adjust the last level to ensure we use exactly count particles
    particleCount[maxDepth] += (count - cumulativeCount);
    
    // Determine which depth level this particle belongs to
    let depth = 0;
    let levelStartIndex = 0;
    
    while (depth < maxDepth && i >= levelStartIndex + particleCount[depth]) {
        levelStartIndex += particleCount[depth];
        depth++;
    }
    
    // Calculate the relative index within this depth level
    const indexInLevel = i - levelStartIndex;
    const levelCount = particleCount[depth];
    
    // Calculate position parameters
    const t = indexInLevel / (levelCount || 1); // Normalized position in level
    
    // For the trunk (depth 0)
    if (depth === 0) {
        // Simple line for the trunk
        return new THREE.Vector3(
            (Math.random() * 2 - 1) * 0.5, // Small random spread for thickness
            -trunkLength / 2 + t * trunkLength,
            (Math.random() * 2 - 1) * 0.5  // Small random spread for thickness
        );
    }
    
    // For branches at higher depths
    // Determine which branch in the current depth
    const branchCount = Math.pow(2, depth);
    const branchIndex = Math.floor(t * branchCount) % branchCount;
    const positionInBranch = (t * branchCount) % 1;
    
    // Calculate the position based on branch path
    let x = 0, y = trunkLength / 2, z = 0; // Start at top of trunk
    let currentLength = trunkLength;
    let currentAngle = 0;
    
    // For the first depth level (branching from trunk)
    if (depth >= 1) {
        currentLength *= branchRatio;
        // Determine left or right branch
        currentAngle = (branchIndex % 2 === 0) ? branchAngle : -branchAngle;
        
        // Move up the branch
        x += Math.sin(currentAngle) * currentLength * positionInBranch;
        y += Math.cos(currentAngle) * currentLength * positionInBranch;
    }
    
    // For higher depths, calculate the full path
    for (let d = 2; d <= depth; d++) {
        currentLength *= branchRatio;
        
        // Determine branch direction at this depth
        // Use bit operations to determine left vs right at each branch
        const pathBit = (branchIndex >> (depth - d)) & 1;
        const nextAngle = pathBit === 0 ? branchAngle : -branchAngle;
        
        // Only apply movement for the branches we've completed
        if (d < depth) {
            // Rotate the current direction and move full branch length
            currentAngle += nextAngle;
            x += Math.sin(currentAngle) * currentLength;
            y += Math.cos(currentAngle) * currentLength;
        } else {
            // For the final branch, move partially based on positionInBranch
            currentAngle += nextAngle;
            x += Math.sin(currentAngle) * currentLength * positionInBranch;
            y += Math.cos(currentAngle) * currentLength * positionInBranch;
        }
    }
    
    // Add small random offsets for volume
    const randomSpread = 0.8 * (1 - Math.pow(branchRatio, depth));
    x += (Math.random() * 2 - 1) * randomSpread;
    z += (Math.random() * 2 - 1) * randomSpread;
    
    return new THREE.Vector3(x, y, z);
}

function createVoronoi(i, count) {
    // Voronoi parameters
    const radius = 30;            // Maximum radius of the sphere to place points on
    const numSites = 25;          // Number of Voronoi sites (cells)
    const cellThickness = 2.5;    // Thickness of the cell boundaries
    const jitter = 0.5;           // Random jitter to make edges look more natural
    
    // First, we generate fixed pseudorandom Voronoi sites (cell centers)
    // We use a deterministic approach to ensure sites are the same for each call
    const sites = [];
    for (let s = 0; s < numSites; s++) {
        // Use a specific seed formula for each site to ensure repeatability
        const seed1 = Math.sin(s * 42.5) * 10000;
        const seed2 = Math.cos(s * 15.3) * 10000;
        const seed3 = Math.sin(s * 33.7) * 10000;
        
        // Generate points on a sphere using spherical coordinates
        const theta = 2 * Math.PI * (seed1 - Math.floor(seed1));
        const phi = Math.acos(2 * (seed2 - Math.floor(seed2)) - 1);
        
        sites.push(new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta) * radius,
            Math.sin(phi) * Math.sin(theta) * radius,
            Math.cos(phi) * radius
        ));
    }
    
    // Now we generate points that lie primarily along the boundaries between Voronoi cells
    
    // First, decide if this is a site point (center of a cell) or a boundary point
    const sitePoints = Math.min(numSites, Math.floor(count * 0.1)); // 10% of points are sites
    
    if (i < sitePoints) {
        // Place this point at a Voronoi site center
        const siteIndex = i % sites.length;
        const site = sites[siteIndex];
        
        // Return the site position with small random variation
        return new THREE.Vector3(
            site.x + (Math.random() * 2 - 1) * jitter,
            site.y + (Math.random() * 2 - 1) * jitter,
            site.z + (Math.random() * 2 - 1) * jitter
        );
    } else {
        // This is a boundary point
        // Generate a random point on the sphere
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        
        const point = new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta) * radius,
            Math.sin(phi) * Math.sin(theta) * radius,
            Math.cos(phi) * radius
        );
        
        // Find the two closest sites to this point
        let closestDist = Infinity;
        let secondClosestDist = Infinity;
        let closestSite = null;
        let secondClosestSite = null;
        
        for (const site of sites) {
            const dist = point.distanceTo(site);
            
            if (dist < closestDist) {
                secondClosestDist = closestDist;
                secondClosestSite = closestSite;
                closestDist = dist;
                closestSite = site;
            } else if (dist < secondClosestDist) {
                secondClosestDist = dist;
                secondClosestSite = site;
            }
        }
        
        // Check if this point is near the boundary between the two closest cells
        const distDiff = Math.abs(closestDist - secondClosestDist);
        
        if (distDiff < cellThickness) {
            // This point is on a boundary
            
            // Add small random jitter to make the boundary look more natural
            point.x += (Math.random() * 2 - 1) * jitter;
            point.y += (Math.random() * 2 - 1) * jitter;
            point.z += (Math.random() * 2 - 1) * jitter;
            
            // Project the point back onto the sphere
            point.normalize().multiplyScalar(radius);
            
            return point;
        } else {
            // Not a boundary point, retry with a different approach
            // Move the point slightly toward the boundary
            const midpoint = new THREE.Vector3().addVectors(closestSite, secondClosestSite).multiplyScalar(0.5);
            const dirToMid = new THREE.Vector3().subVectors(midpoint, point).normalize();
            
            // Move point toward the midpoint between cells
            point.add(dirToMid.multiplyScalar(distDiff * 0.7));
            
            // Add small random jitter
            point.x += (Math.random() * 2 - 1) * jitter;
            point.y += (Math.random() * 2 - 1) * jitter;
            point.z += (Math.random() * 2 - 1) * jitter;
            
            // Project back onto the sphere
            point.normalize().multiplyScalar(radius);
            
            return point;
        }
    }
}

// ==================== 新增图案 ====================

function createAudiLogo(i, count) {
    // 奥迪四环标志
    const ringRadius = 12;    // 每个环的半径
    const tubeRadius = 2;     // 管道粗细
    const ringSpacing = 18;   // 环中心间距
    const numRings = 4;
    const particlesPerRing = Math.floor(count / numRings);
    const ringIndex = Math.floor(i / particlesPerRing);
    const indexInRing = i % particlesPerRing;
    
    // 每个环的中心X坐标（从左到右排列）
    const ringCenterX = (ringIndex - (numRings - 1) / 2) * ringSpacing;
    
    // 在环面上均匀分布粒子
    const u = (indexInRing / particlesPerRing) * Math.PI * 2;
    const v = ((indexInRing * 7.13) % 1) * Math.PI * 2; // 伪随机分布
    
    const x = ringCenterX + (ringRadius + tubeRadius * Math.cos(v)) * Math.cos(u);
    const y = (ringRadius + tubeRadius * Math.cos(v)) * Math.sin(u);
    const z = tubeRadius * Math.sin(v);
    
    return new THREE.Vector3(x, y, z);
}

function createDNA(i, count) {
    // DNA 双螺旋
    const numStrands = 2;
    const strandIndex = i % numStrands;
    const t = Math.floor(i / numStrands) / Math.floor(count / numStrands);
    const height = 80;
    const radius = 15;
    const rotations = 5;
    const angleOffset = strandIndex * Math.PI;
    const angle = t * Math.PI * 2 * rotations + angleOffset;
    const y = (t - 0.5) * height;
    
    // 判断是否为连接桥粒子（20%的粒子用于连接两条链）
    const bridgeRatio = 0.2;
    const isBridge = (i % 5 === 0) && strandIndex === 0;
    
    if (isBridge) {
        // 连接桥：在两条链之间
        const bridgeT = Math.random();
        const otherAngle = angle + Math.PI; // 另一条链的角度
        const x1 = Math.cos(angle) * radius;
        const z1 = Math.sin(angle) * radius;
        const x2 = Math.cos(otherAngle) * radius;
        const z2 = Math.sin(otherAngle) * radius;
        return new THREE.Vector3(
            x1 + (x2 - x1) * bridgeT,
            y,
            z1 + (z2 - z1) * bridgeT
        );
    }
    
    return new THREE.Vector3(
        Math.cos(angle) * radius,
        y,
        Math.sin(angle) * radius
    );
}

function createHeart(i, count) {
    // 3D 心形
    const scale = 1.8;
    const t = i / count;
    
    // 心形参数方程
    const u = t * Math.PI * 2;
    const v = ((i * 3.57) % 1) * Math.PI; // 纬度方向分布
    
    // 心形轮廓（2D心形曲线绕Y轴旋转形成3D）
    const heartR = 16 * Math.pow(Math.sin(u), 3);
    const heartY = 13 * Math.cos(u) - 5 * Math.cos(2*u) - 2 * Math.cos(3*u) - Math.cos(4*u);
    
    // 添加厚度，形成3D效果
    const thickness = 3;
    const x = heartR * scale * Math.sin(v) / 16 + (Math.random() * 2 - 1) * thickness * Math.abs(Math.sin(v));
    const y = heartY * scale + (Math.random() * 2 - 1) * thickness * 0.3;
    const z = heartR * scale * Math.cos(v) / 16 + (Math.random() * 2 - 1) * thickness * Math.abs(Math.cos(v));
    
    return new THREE.Vector3(x, y, z);
}

function createOlympicRings(i, count) {
    // 奥运五环
    const ringRadius = 12;
    const tubeRadius = 1.8;
    const hSpacing = 20;  // 水平间距
    const vSpacing = 10;  // 垂直间距
    const numRings = 5;
    const particlesPerRing = Math.floor(count / numRings);
    const ringIndex = Math.floor(i / particlesPerRing);
    const indexInRing = i % particlesPerRing;
    
    // 五环布局：上排3个，下排2个
    let ringCenterX, ringCenterY;
    if (ringIndex < 3) {
        // 上排：蓝、黑、红
        ringCenterX = (ringIndex - 1) * hSpacing;
        ringCenterY = vSpacing / 2;
    } else {
        // 下排：黄、绿
        ringCenterX = (ringIndex === 3 ? -0.5 : 0.5) * hSpacing;
        ringCenterY = -vSpacing / 2;
    }
    
    const u = (indexInRing / particlesPerRing) * Math.PI * 2;
    const v = ((indexInRing * 7.13) % 1) * Math.PI * 2;
    
    const x = ringCenterX + (ringRadius + tubeRadius * Math.cos(v)) * Math.cos(u);
    const y = ringCenterY + (ringRadius + tubeRadius * Math.cos(v)) * Math.sin(u);
    const z = tubeRadius * Math.sin(v);
    
    return new THREE.Vector3(x, y, z);
}

function createStar(i, count) {
    // 五角星
    const outerRadius = 35;
    const innerRadius = 14;
    const numPoints = 5;
    const thickness = 3;
    
    // 分配粒子：50%在星形轮廓，50%在填充面
    const outlineCount = Math.floor(count * 0.5);
    const isOutline = i < outlineCount;
    
    if (isOutline) {
        // 轮廓粒子：沿五角星边缘分布
        const totalSegments = numPoints * 2; // 10条边
        const segmentIndex = i % totalSegments;
        const t = (i / outlineCount) * totalSegments;
        const segIdx = Math.floor(t) % totalSegments;
        const segT = t - Math.floor(t);
        
        // 计算当前边的起点和终点
        const startAngle = (segIdx / totalSegments) * Math.PI * 2 - Math.PI / 2;
        const endAngle = ((segIdx + 1) / totalSegments) * Math.PI * 2 - Math.PI / 2;
        const startR = segIdx % 2 === 0 ? outerRadius : innerRadius;
        const endR = segIdx % 2 === 0 ? innerRadius : outerRadius;
        
        const x = (Math.cos(startAngle) * startR * (1 - segT) + Math.cos(endAngle) * endR * segT);
        const y = (Math.sin(startAngle) * startR * (1 - segT) + Math.sin(endAngle) * endR * segT);
        const z = (Math.random() * 2 - 1) * thickness;
        
        return new THREE.Vector3(x, y, z);
    } else {
        // 填充粒子：在五角星内部随机分布
        const fillIndex = i - outlineCount;
        const fillCount = count - outlineCount;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * outerRadius;
        
        // 检查点是否在星形内部（简化：使用径向裁剪）
        const starAngle = ((angle + Math.PI / 2) % (Math.PI * 2)) / (Math.PI * 2) * numPoints * 2;
        const segment = Math.floor(starAngle) % (numPoints * 2);
        const maxR = segment % 2 === 0 ? outerRadius : innerRadius;
        const actualR = r * maxR / outerRadius;
        
        return new THREE.Vector3(
            Math.cos(angle) * actualR + (Math.random() * 2 - 1) * 1,
            Math.sin(angle) * actualR + (Math.random() * 2 - 1) * 1,
            (Math.random() * 2 - 1) * thickness
        );
    }
}

function createButterfly(i, count) {
    // 蝴蝶曲线
    const t = (i / count) * Math.PI * 12;
    const scale = 8;
    
    // 蝴蝶曲线参数方程
    const r = Math.exp(Math.sin(t)) - 2 * Math.cos(4 * t) + Math.pow(Math.sin((2 * t - Math.PI) / 24), 5);
    
    const x = Math.sin(t) * r * scale;
    const y = Math.cos(t) * r * scale;
    const z = Math.sin(t * 0.5) * 5 + (Math.random() * 2 - 1) * 2; // 轻微3D效果
    
    return new THREE.Vector3(x, y, z);
}

function createTornado(i, count) {
    // 龙卷风
    const t = i / count;
    const height = 80;
    const bottomRadius = 3;
    const topRadius = 40;
    const rotations = 8;
    
    const y = (t - 0.5) * height;
    const currentRadius = bottomRadius + (topRadius - bottomRadius) * t;
    const angle = t * Math.PI * 2 * rotations;
    
    // 添加湍流效果
    const turbulence = (1 - t) * 3;
    const x = Math.cos(angle) * currentRadius + (Math.random() * 2 - 1) * turbulence;
    const z = Math.sin(angle) * currentRadius + (Math.random() * 2 - 1) * turbulence;
    
    return new THREE.Vector3(x, y, z);
}

function createAtom(i, count) {
    // 原子模型：原子核 + 电子轨道
    const nucleusCount = Math.floor(count * 0.15);
    const numOrbits = 3;
    const orbitParticles = Math.floor((count - nucleusCount) / numOrbits);
    
    if (i < nucleusCount) {
        // 原子核：密集球体
        const t = i / nucleusCount;
        const phi = Math.acos(2 * t - 1);
        const theta = 2 * Math.PI * i * (1 + Math.sqrt(5));
        const r = 5 + Math.random() * 2;
        return new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta) * r,
            Math.sin(phi) * Math.sin(theta) * r,
            Math.cos(phi) * r
        );
    }
    
    // 电子轨道
    const orbitIndex = Math.floor((i - nucleusCount) / orbitParticles);
    const indexInOrbit = (i - nucleusCount) % orbitParticles;
    const t = indexInOrbit / orbitParticles;
    const angle = t * Math.PI * 2;
    const orbitRadius = 25;
    
    // 三个轨道分别在不同平面上
    const tiltAngles = [0, Math.PI / 3, -Math.PI / 3];
    const rotationAngles = [0, Math.PI / 3, -Math.PI / 3];
    const tilt = tiltAngles[orbitIndex % numOrbits];
    const rotation = rotationAngles[orbitIndex % numOrbits];
    
    let x = Math.cos(angle) * orbitRadius;
    let y = Math.sin(angle) * orbitRadius;
    let z = (Math.random() * 2 - 1) * 1.5; // 轨道厚度
    
    // 绕X轴倾斜
    const y2 = y * Math.cos(tilt) - z * Math.sin(tilt);
    const z2 = y * Math.sin(tilt) + z * Math.cos(tilt);
    
    // 绕Y轴旋转
    const x3 = x * Math.cos(rotation) + z2 * Math.sin(rotation);
    const z3 = -x * Math.sin(rotation) + z2 * Math.cos(rotation);
    
    return new THREE.Vector3(x3, y2, z3);
}

function createSharingan(i, count) {
    // 写轮眼：红色虹膜 + 黑色瞳孔 + 3个巴纹(tomoe) + 连接环
    // 正面朝向观察者，XY平面

    const irisCount = Math.floor(count * 0.40);      // 红色虹膜（外圈圆盘）
    const pupilCount = Math.floor(count * 0.12);     // 黑色瞳孔（中心小圆）
    const ringCount = Math.floor(count * 0.15);      // 连接环（瞳孔与巴纹之间的圆环）
    const tomoeCount = Math.floor(count * 0.33);     // 3个巴纹

    const irisR = 30;     // 虹膜半径
    const pupilR = 8;     // 瞳孔半径
    const ringR = 16;     // 连接环半径
    const ringWidth = 1.5;// 连接环宽度
    const tomoeR = 7;     // 巴纹主体半径
    const tomoeDist = 16; // 巴纹中心到瞳孔中心的距离

    if (i < irisCount) {
        // 红色虹膜：大圆盘，排除瞳孔区域
        const angle = Math.random() * Math.PI * 2;
        const r = pupilR + Math.random() * (irisR - pupilR);
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        const z = (Math.random() * 2 - 1) * 0.8;
        return new THREE.Vector3(x, y, z);

    } else if (i < irisCount + pupilCount) {
        // 黑色瞳孔：中心小圆
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * pupilR;
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        const z = (Math.random() * 2 - 1) * 0.8;
        return new THREE.Vector3(x, y, z);

    } else if (i < irisCount + pupilCount + ringCount) {
        // 连接环：瞳孔周围的细圆环
        const angle = Math.random() * Math.PI * 2;
        const r = ringR + (Math.random() * 2 - 1) * ringWidth;
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        const z = (Math.random() * 2 - 1) * 0.8;
        return new THREE.Vector3(x, y, z);

    } else {
        // 3个巴纹(tomoe)：逗号形状，每个由圆形头部+渐细尾巴组成
        const idx = i - irisCount - pupilCount - ringCount;
        const tomoeIdx = idx % 3;
        // 三个巴纹均匀分布，间隔120°
        const tomoeAngle = tomoeIdx * Math.PI * 2 / 3 - Math.PI / 2;
        // 巴纹中心位置
        const cx = Math.cos(tomoeAngle) * tomoeDist;
        const cy = Math.sin(tomoeAngle) * tomoeDist;

        const localIdx = Math.floor(idx / 3);
        const localCount = Math.floor(tomoeCount / 3);

        // 巴纹：70%头部圆形 + 30%尾巴弧线
        if (Math.random() < 0.7) {
            // 圆形头部
            const a = Math.random() * Math.PI * 2;
            const r = Math.random() * tomoeR;
            const x = cx + Math.cos(a) * r;
            const y = cy + Math.sin(a) * r;
            const z = (Math.random() * 2 - 1) * 0.8;
            return new THREE.Vector3(x, y, z);
        } else {
            // 尾巴：从头部沿圆弧向逆时针方向渐细延伸
            const t = Math.random();
            // 尾巴沿连接环的弧线延伸，从巴纹头部位置开始
            const tailStartAngle = tomoeAngle;
            // 逆时针延伸约60°
            const tailEndAngle = tomoeAngle + Math.PI / 3;
            const tailAngle = tailStartAngle + t * (tailEndAngle - tailStartAngle);
            // 尾巴从tomoeR宽度渐细到0
            const tailWidth = tomoeR * (1 - t) * 0.8;
            // 尾巴在连接环弧线上
            const tailR = tomoeDist;
            const baseX = Math.cos(tailAngle) * tailR;
            const baseY = Math.sin(tailAngle) * tailR;
            // 垂直于弧线方向的偏移（渐细）
            const perpAngle = tailAngle + Math.PI / 2;
            const offset = (Math.random() * 2 - 1) * tailWidth;
            const x = baseX + Math.cos(perpAngle) * offset;
            const y = baseY + Math.sin(perpAngle) * offset;
            const z = (Math.random() * 2 - 1) * 0.8;
            return new THREE.Vector3(x, y, z);
        }
    }
}

function createCar(i, count) {
    // 轿车侧视图轮廓：引擎盖、前挡风、车顶、后窗、后备箱、车身、轮子、细节
    // X=前后方向（正=前），Y=上下，Z=左右
    const bodyCount = Math.floor(count * 0.22);       // 车身主体
    const hoodCount = Math.floor(count * 0.08);       // 引擎盖
    const roofCount = Math.floor(count * 0.08);       // 车顶
    const windshieldCount = Math.floor(count * 0.06); // 前挡风玻璃
    const rearWindowCount = Math.floor(count * 0.05); // 后窗
    const trunkCount = Math.floor(count * 0.06);      // 后备箱
    const wheelCount = Math.floor(count * 0.18);      // 4个轮子
    const bumperCount = Math.floor(count * 0.06);     // 前后保险杠
    const lightCount = Math.floor(count * 0.05);      // 车灯
    const doorCount = Math.floor(count * 0.06);       // 车门轮廓
    const mirrorCount = Math.floor(count * 0.03);     // 后视镜
    const grilleCount = Math.floor(count * 0.04);     // 前格栅
    const exhaustCount = Math.floor(count * 0.03);    // 排气管

    if (i < bodyCount) {
        // 车身主体：梯形截面，下宽上窄
        const x = (Math.random() * 2 - 1) * 28;
        const yBase = 2;
        const yTop = 10;
        const y = yBase + Math.random() * (yTop - yBase);
        // 侧面有弧度：越靠近两端越窄
        const xNorm = Math.abs(x) / 28;
        const zHalf = 8 * (1 - xNorm * 0.15); // 两端微收
        const z = (Math.random() * 2 - 1) * zHalf;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount) {
        // 引擎盖：前部低矮斜面
        const x = 12 + Math.random() * 18; // 前半部分
        const y = 8 + (30 - x) * 0.1 + (Math.random() * 2 - 1) * 0.5; // 微微前倾
        const z = (Math.random() * 2 - 1) * 7.5;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount) {
        // 车顶：弧形顶棚
        const x = -6 + (Math.random() * 2 - 1) * 12; // 中部偏后
        const y = 16 + Math.random() * 2; // 高位
        const z = (Math.random() * 2 - 1) * 6.5;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount) {
        // 前挡风玻璃：倾斜面
        const t = Math.random();
        const x = 4 + t * 10; // 从车顶前缘到引擎盖后端
        const y = 10 + t * 6; // 从低到高
        const z = (Math.random() * 2 - 1) * 6;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount) {
        // 后窗：倾斜面
        const t = Math.random();
        const x = -6 - t * 10; // 从车顶后缘到后备箱前端
        const y = 10 + (1 - t) * 6; // 从高到低
        const z = (Math.random() * 2 - 1) * 6;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount) {
        // 后备箱：后部低矮
        const x = -14 - Math.random() * 14;
        const y = 7 + (x + 28) * 0.05 + (Math.random() * 2 - 1) * 0.5;
        const z = (Math.random() * 2 - 1) * 7;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount) {
        // 4个轮子：轮胎+轮毂
        const idx = i - bodyCount - hoodCount - roofCount - windshieldCount - rearWindowCount - trunkCount;
        const wheelIdx = idx % 4;
        const particlesPerWheel = Math.floor(wheelCount / 4);
        const idxInWheel = idx % particlesPerWheel;
        // 轮子位置：前轮x=16，后轮x=-16，左右两侧
        const wx = wheelIdx < 2 ? 16 : -16;
        const side = wheelIdx % 2 === 0 ? 1 : -1;
        const wz = side * 8.5;
        const wheelR = 5;
        const tireW = 2.5;
        const angle = (idxInWheel / particlesPerWheel) * Math.PI * 2;
        // 轮胎外圈
        const isTire = Math.random() > 0.25;
        const r = isTire ? wheelR + (Math.random() * 2 - 1) * 0.5 : Math.random() * (wheelR - 1);
        return new THREE.Vector3(
            wx + (Math.random() * 2 - 1) * tireW / 2,
            r - 3, // 轮子底部接近地面
            wz + Math.cos(angle) * tireW / 2
        );

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount + bumperCount) {
        // 前后保险杠
        const idx = i - bodyCount - hoodCount - roofCount - windshieldCount - rearWindowCount - trunkCount - wheelCount;
        const isFront = idx < bumperCount / 2;
        const x = isFront ? 28 + (Math.random() * 2 - 1) * 2 : -28 + (Math.random() * 2 - 1) * 2;
        const y = 2 + Math.random() * 5;
        const z = (Math.random() * 2 - 1) * 8;
        return new THREE.Vector3(x, y, z);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount + bumperCount + lightCount) {
        // 车灯：前大灯+后尾灯
        const idx = i - bodyCount - hoodCount - roofCount - windshieldCount - rearWindowCount - trunkCount - wheelCount - bumperCount;
        const isFront = idx < lightCount / 2;
        if (isFront) {
            // 前大灯：两侧各一个
            const side = Math.random() > 0.5 ? 1 : -1;
            return new THREE.Vector3(27 + Math.random() * 2, 6 + Math.random() * 3, side * (5 + Math.random() * 2));
        } else {
            // 后尾灯：两侧各一个
            const side = Math.random() > 0.5 ? 1 : -1;
            return new THREE.Vector3(-27 - Math.random() * 2, 6 + Math.random() * 3, side * (5 + Math.random() * 2));
        }

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount + bumperCount + lightCount + doorCount) {
        // 车门轮廓：两侧各两个门的边框线
        const idx = i - bodyCount - hoodCount - roofCount - windshieldCount - rearWindowCount - trunkCount - wheelCount - bumperCount - lightCount;
        const side = idx % 2 === 0 ? 1 : -1;
        const doorIdx = idx % 4 < 2 ? 0 : 1; // 前门/后门
        const t = Math.random();
        // 门框：矩形轮廓
        const doorX = doorIdx === 0 ? (2 + t * 12) : (-10 + t * 12);
        const doorY = 3 + Math.random() * 8;
        // 只在边缘（门框线）
        const isEdge = Math.random() > 0.5;
        const doorZ = isEdge ? side * 8.2 : side * (7 + Math.random());
        return new THREE.Vector3(doorX, doorY, doorZ);

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount + bumperCount + lightCount + doorCount + mirrorCount) {
        // 后视镜：两侧各一个
        const side = Math.random() > 0.5 ? 1 : -1;
        return new THREE.Vector3(8, 10 + (Math.random() * 2 - 1) * 1.5, side * (9 + Math.random() * 2));

    } else if (i < bodyCount + hoodCount + roofCount + windshieldCount + rearWindowCount + trunkCount + wheelCount + bumperCount + lightCount + doorCount + mirrorCount + grilleCount) {
        // 前格栅：车头中央竖条
        const y = 3 + Math.random() * 4;
        const z = (Math.random() * 2 - 1) * 5;
        return new THREE.Vector3(29 + Math.random(), y, z);

    } else {
        // 排气管：车尾底部
        const side = Math.random() > 0.5 ? 1 : -1;
        return new THREE.Vector3(-29 - Math.random() * 3, 0 + Math.random() * 2, side * (3 + Math.random() * 2));
    }
}

function createBullet(i, count) {
    // 步枪子弹：圆弧弹头(ogive) + 瓶颈弹壳 + 底火
    // 参考典型7.62mm步枪弹比例
    // 子弹朝上，Y轴正方向为弹头方向

    const ogiveCount = Math.floor(count * 0.12);      // 圆弧弹头尖
    const bearingCount = Math.floor(count * 0.08);     // 弹头圆柱段
    const neckCount = Math.floor(count * 0.10);        // 弹壳颈部（窄）
    const shoulderCount = Math.floor(count * 0.12);    // 弹壳肩部（瓶颈过渡）
    const bodyCount = Math.floor(count * 0.38);        // 弹壳主体（粗）
    const rimCount = Math.floor(count * 0.08);         // 弹壳底缘
    const primerCount = Math.floor(count * 0.12);      // 底火

    const c1 = ogiveCount;
    const c2 = c1 + bearingCount;
    const c3 = c2 + neckCount;
    const c4 = c3 + shoulderCount;
    const c5 = c4 + bodyCount;
    const c6 = c5 + rimCount;

    if (i < c1) {
        // 圆弧弹头尖 (ogive curve) — 使用正弦曲线模拟圆弧过渡，非简单锥形
        const t = i / c1;
        const tipHeight = 12;
        const maxR = 4;
        const y = 28 + t * tipHeight;
        // ogive曲线：半径按弧线增长，尖端圆滑
        const ogiveT = Math.sin(t * Math.PI * 0.5);
        const currentR = maxR * ogiveT;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * currentR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else if (i < c2) {
        // 弹头圆柱段 (bearing surface) — 弹头中段平直部分
        const bearingR = 4;
        const bearingHeight = 6;
        const y = 22 + Math.random() * bearingHeight;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() < 0.7 ? bearingR + (Math.random() * 2 - 1) * 0.3 : Math.random() * bearingR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else if (i < c3) {
        // 弹壳颈部 (case neck) — 窄圆柱，包裹弹头底部
        const neckR = 4.5;
        const neckHeight = 8;
        const y = 14 + Math.random() * neckHeight;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() < 0.7 ? neckR + (Math.random() * 2 - 1) * 0.3 : Math.random() * neckR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else if (i < c4) {
        // 弹壳肩部 (case shoulder) — 瓶颈过渡：从颈部窄到主体粗
        const t = (i - c3) / shoulderCount;
        const shoulderHeight = 8;
        const neckR = 4.5;
        const bodyR = 7;
        const y = 6 + t * shoulderHeight;
        const currentR = neckR + t * (bodyR - neckR);
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() < 0.7 ? currentR + (Math.random() * 2 - 1) * 0.3 : Math.random() * currentR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else if (i < c5) {
        // 弹壳主体 (case body) — 较粗的圆柱，明显比颈部宽
        const bodyR = 7;
        const bodyHeight = 24;
        const y = -18 + Math.random() * bodyHeight;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() < 0.7 ? bodyR + (Math.random() * 2 - 1) * 0.3 : Math.random() * bodyR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else if (i < c6) {
        // 弹壳底缘 (case rim) — 底部略外扩的环
        const rimR = 7.8;
        const y = -18 - Math.random() * 2;
        const angle = Math.random() * Math.PI * 2;
        const r = rimR + (Math.random() * 2 - 1) * 0.5;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    } else {
        // 底火 (primer) — 底部中央圆形凹槽
        const primerR = 3;
        const y = -20 - Math.random() * 1.5;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * primerR;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            y,
            Math.sin(angle) * r
        );
    }
}

function createRocket(i, count) {
    // 火箭：箭头 + 箭体 + 尾翼 + 火焰
    const noseCount = Math.floor(count * 0.15);
    const bodyCount = Math.floor(count * 0.40);
    const finCount = Math.floor(count * 0.15);
    // 剩余为火焰

    if (i < noseCount) {
        // 箭头：锥形
        const t = i / noseCount;
        const noseHeight = 20;
        const maxR = 7;
        const y = 30 + t * noseHeight;
        const currentR = maxR * (1 - t);
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * currentR;
        return new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r);
    } else if (i < noseCount + bodyCount) {
        // 箭体：圆柱
        const idx = i - noseCount;
        const t = idx / bodyCount;
        const bodyHeight = 40;
        const bodyR = 7;
        const y = -10 + t * bodyHeight;
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() < 0.6 ? bodyR + (Math.random() * 2 - 1) * 0.5 : Math.random() * bodyR;
        return new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r);
    } else if (i < noseCount + bodyCount + finCount) {
        // 尾翼：4片对称翼
        const idx = i - noseCount - bodyCount;
        const finIdx = idx % 4;
        const finAngle = finIdx * Math.PI / 2;
        const finHeight = 12;
        const finLength = 12;
        const t = (idx % Math.floor(finCount / 4)) / Math.floor(finCount / 4);
        const h = t * finHeight;
        const l = Math.random() * finLength;
        const baseR = 7;
        return new THREE.Vector3(
            Math.cos(finAngle) * (baseR + l) + (Math.random() * 2 - 1) * 0.5,
            -10 - h + Math.random() * 2,
            Math.sin(finAngle) * (baseR + l) + (Math.random() * 2 - 1) * 0.5
        );
    } else {
        // 火焰：锥形散射
        const idx = i - noseCount - bodyCount - finCount;
        const flameCount = count - noseCount - bodyCount - finCount;
        const t = idx / flameCount;
        const flameLength = 20;
        const y = -10 - t * flameLength;
        const maxR = 8 * t; // 越远越散
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * maxR;
        return new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r);
    }
}

function createSword(i, count) {
    // 宝剑：剑身 + 剑柄 + 护手
    const bladeCount = Math.floor(count * 0.55);
    const guardCount = Math.floor(count * 0.10);
    // 剩余为剑柄

    if (i < bladeCount) {
        // 剑身：扁平菱形截面的长条
        const t = i / bladeCount;
        const bladeLength = 60;
        const maxWidth = 5;
        const thickness = 1.5;
        const y = 10 + t * bladeLength;
        // 剑尖收窄
        const widthFactor = t < 0.85 ? 1.0 : (1.0 - (t - 0.85) / 0.15);
        const currentWidth = maxWidth * widthFactor;
        const x = (Math.random() * 2 - 1) * currentWidth;
        const z = (Math.random() * 2 - 1) * thickness;
        return new THREE.Vector3(x, y, z);
    } else if (i < bladeCount + guardCount) {
        // 护手：横向扁条
        const guardWidth = 16;
        const x = (Math.random() * 2 - 1) * guardWidth;
        const y = 8 + (Math.random() * 2 - 1) * 2;
        const z = (Math.random() * 2 - 1) * 2;
        return new THREE.Vector3(x, y, z);
    } else {
        // 剑柄：圆柱 + 圆球末端
        const idx = i - bladeCount - guardCount;
        const handleCount = count - bladeCount - guardCount;
        const handleLength = 18;
        const handleR = 2.5;
        // 80%柄身，20%柄尾圆球
        if (idx < handleCount * 0.8) {
            const t = idx / (handleCount * 0.8);
            const y = -handleLength * t + 6;
            const angle = Math.random() * Math.PI * 2;
            const r = handleR + (Math.random() * 2 - 1) * 0.3;
            return new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r);
        } else {
            // 柄尾圆球
            const phi = Math.acos(2 * Math.random() - 1);
            const theta = Math.random() * Math.PI * 2;
            const r = 4;
            return new THREE.Vector3(
                Math.sin(phi) * Math.cos(theta) * r,
                -handleLength + 4 + Math.cos(phi) * r,
                Math.sin(phi) * Math.sin(theta) * r
            );
        }
    }
}

function createCrown(i, count) {
    // 皇冠：底环 + 尖齿 + 顶部圆球
    const baseCount = Math.floor(count * 0.35);
    const pointCount = Math.floor(count * 0.45);
    // 剩余为顶部圆球

    if (i < baseCount) {
        // 底部环带
        const t = i / baseCount;
        const angle = t * Math.PI * 2;
        const radius = 25;
        const bandHeight = 8;
        const r = radius + (Math.random() * 2 - 1) * 2;
        return new THREE.Vector3(
            Math.cos(angle) * r,
            (Math.random() * 2 - 1) * bandHeight / 2 - 10,
            Math.sin(angle) * r
        );
    } else if (i < baseCount + pointCount) {
        // 尖齿：5个三角形尖齿
        const idx = i - baseCount;
        const numPoints = 5;
        const pointIdx = idx % numPoints;
        const particlesPerPoint = Math.floor(pointCount / numPoints);
        const idxInPoint = idx % particlesPerPoint;
        const t = idxInPoint / particlesPerPoint;
        const pointAngle = (pointIdx / numPoints) * Math.PI * 2;
        const baseRadius = 25;
        const pointHeight = 25;
        // 三角形：底边在环上，顶点在上方
        const h = t * pointHeight;
        const currentR = baseRadius * (1 - t * 0.3); // 微微内收
        const angleOffset = (Math.random() * 2 - 1) * 0.15;
        return new THREE.Vector3(
            Math.cos(pointAngle + angleOffset) * currentR,
            -6 + h,
            Math.sin(pointAngle + angleOffset) * currentR
        );
    } else {
        // 顶部圆球装饰
        const idx = i - baseCount - pointCount;
        const numPoints = 5;
        const ballIdx = idx % numPoints;
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = Math.random() * Math.PI * 2;
        const ballR = 3;
        const pointAngle = (ballIdx / numPoints) * Math.PI * 2;
        const baseRadius = 25 * 0.7;
        return new THREE.Vector3(
            Math.cos(pointAngle) * baseRadius + Math.sin(phi) * Math.cos(theta) * ballR,
            19 + Math.cos(phi) * ballR,
            Math.sin(pointAngle) * baseRadius + Math.sin(phi) * Math.sin(theta) * ballR
        );
    }
}